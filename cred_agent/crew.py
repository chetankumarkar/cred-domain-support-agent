"""Mock-first crew orchestration, memory, schema, budget, and response cache."""
from __future__ import annotations
import os, re, json
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY","true"); os.environ.setdefault("OTEL_SDK_DISABLED","true")
os.environ["CREWAI_TRACING_ENABLED"]="false"
os.environ.setdefault("MOCK_LLM","1")
from collections import defaultdict
from pydantic import BaseModel, Field
from .rag import grounded_answer
from .dataset import check_loan_application_status
from .guardrails import validate_input, groundedness_check, GuardrailError

class SupportResponse(BaseModel):
    answer: str
    intent: str
    sources: list[str] = Field(default_factory=list)
    application: dict | None = None
    escalated: bool = False
    trace_id: str = ""

class RequestBudgetError(ValueError): pass
class SessionMemory:
    def __init__(self):
        self.messages=defaultdict(list)
        try:
            from langchain_core.chat_history import InMemoryChatMessageHistory
            self._history_type=InMemoryChatMessageHistory
        except ImportError: self._history_type=None
    def add(self,sid,user,assistant):
        if self._history_type:
            if sid not in self.messages: self.messages[sid]=self._history_type()
            self.messages[sid].add_user_message(user); self.messages[sid].add_ai_message(assistant)
        else: self.messages[sid].extend([{"role":"user","content":user},{"role":"assistant","content":assistant}])
    def history(self,sid):
        if sid not in self.messages: return []
        value=self.messages[sid]
        if self._history_type:
            return [{"role":"user" if m.type=="human" else "assistant","content":m.content} for m in value.messages]
        return list(value)
    def clear(self,sid): self.messages.pop(sid,None)

MEMORY=SessionMemory(); CACHE={}; COUNTERS={"rag_calls":0,"lookup_calls":0,"cache_hits":0,"llm_calls":0}
MAX_CHARS=4000; MAX_ESTIMATED_TOKENS=1200
def _budget(text):
    estimate=(len(text)+3)//4
    if len(text)>MAX_CHARS or estimate>MAX_ESTIMATED_TOKENS:
        raise RequestBudgetError(f"Request exceeds budget: {len(text)} chars / {estimate} estimated tokens; limits are {MAX_CHARS} / {MAX_ESTIMATED_TOKENS}.")
def _key(text): return " ".join(text.casefold().split())

def _application_from_messages(messages):
    """Read structured lookup output from CrewAI's tool/context messages."""
    if not isinstance(messages,list): return None
    for message in reversed(messages):
        if not isinstance(message,dict): continue
        content=message.get("content","")
        candidates=[content] if isinstance(content,str) else []
        if isinstance(content,str): candidates.extend(re.findall(r"\{[^{}]*\}",content))
        for candidate in candidates:
            try: value=json.loads(candidate)
            except (json.JSONDecodeError,TypeError): continue
            if isinstance(value,dict) and value.get("record_id","").startswith("CRD-") and "found" in value:
                return value
    return None

def run_support(query: str, session_id: str="demo", trace_id: str="") -> SupportResponse:
    _budget(query)
    safe,meta=validate_input(query)
    text=safe.strip(); is_lookup=bool(re.search(r"\b(CRD-\d{4})\b",text,re.I))
    if os.getenv("USE_CREWAI_CREW","0")=="1":
        key=_key(text)
        if not is_lookup and key in CACHE:
            COUNTERS["cache_hits"]+=1
            result=CACHE[key].model_copy(update={"trace_id":trace_id})
        else:
            result=crewai_kickoff(text,session_id)
            if is_lookup and not result.application:
                raise RuntimeError("CrewAI Lookup Agent returned no structured application result")
            if not is_lookup:
                rag=grounded_answer(text)
                answer=groundedness_check(result.answer,rag.get("context", ""),rag["abstained"])
                result=result.model_copy(update={"answer":answer,"sources":rag["sources"],"trace_id":trace_id})
                CACHE[key]=result.model_copy()
        MEMORY.add(session_id,text,result.answer)
        return result.model_copy(update={"trace_id":trace_id})
    if is_lookup:
        COUNTERS["lookup_calls"]+=1
        data=check_loan_application_status(re.search(r"CRD-\d{4}",text,re.I).group())
        answer=(f"Application {data['record_id']} is {data['status']}; recorded amount INR {data['loan_amount_inr']:,}. " + ("This case should be routed for additional review." if data['escalate'] else "No additional escalation is indicated by this score.")) if data.get("found") else data["message"]
        result=SupportResponse(answer=answer,intent="application_status",application=data if data.get("found") else None,escalated=bool(data.get("escalate")),trace_id=trace_id)
    else:
        key=_key(text)
        cached=CACHE.get(key)
        if cached:
            COUNTERS["cache_hits"]+=1
            result=cached.model_copy(update={"trace_id":trace_id})
        else:
            COUNTERS["rag_calls"]+=1; COUNTERS["llm_calls"]+=1
            answer=grounded_answer(text)
            context=answer.get("context", "")
            final=groundedness_check(answer["answer"],context,answer["abstained"])
            result=SupportResponse(answer=final,intent="policy_question",sources=answer["sources"],trace_id=trace_id)
            CACHE[key]=result.model_copy()
    MEMORY.add(session_id,text,result.answer)
    return result

def crewai_kickoff(query: str, session_id: str="default") -> SupportResponse:
    """Optional native CrewAI implementation. This path is exercised when dependencies are installed.

    MOCK_LLM extends CrewAI's documented BaseLLM extension point. Tool arguments are dispatched
    from the declared JSON schema; the record ID check never depends on tool-name substrings.
    """
    try:
        from crewai import Agent, Crew, Task, Process
        from crewai.llms.base_llm import BaseLLM
        from crewai.tools import tool
    except ImportError as e:
        raise RuntimeError("Install requirements.txt to run the native CrewAI adapter") from e

    class MockLLM(BaseLLM):
        def __init__(self): super().__init__(model="mock/deterministic", temperature=0)
        def supports_function_calling(self): return True
        def call(self, messages, tools=None, callbacks=None, **kwargs):
            COUNTERS["llm_calls"]+=1
            if tools:
                # Native function-calling conversations include a tool-result message
                # after the function executes. Return a final answer at that point;
                # otherwise CrewAI will see the same function call over and over.
                if isinstance(messages, list) and any(
                    isinstance(message, dict)
                    and (message.get("role") in {"tool", "function"} or message.get("tool_call_id"))
                    for message in messages
                ):
                    if re.search(r"\bCRD-\d{4}\b",query,re.I):
                        application=_application_from_messages(messages)
                        if not application: return "No application matched that reference."
                        return json.dumps(application)
                    else:
                        from .rag import grounded_answer
                        answer=grounded_answer(query)["answer"]
                    return answer
                # Select from the declared argument schema. In particular, rag_lookup is
                # never routed by inspecting its name for the word "lookup".
                for spec in tools:
                    fn=spec.get("function",spec)
                    props=fn.get("parameters",{}).get("properties",{})
                    if "query" in props:
                        return [{"type":"function","function":{"name":fn["name"],"arguments":json.dumps({"query":query})}}]
                    if "record_id" in props:
                        found=re.search(r"CRD-\d{4}",query,re.I)
                        return [{"type":"function","function":{"name":fn["name"],"arguments":json.dumps({"record_id":found.group() if found else "CRD-0001"})}}]
            if re.search(r"\bCRD-\d{4}\b",query,re.I):
                application=_application_from_messages(messages)
                if not application or not application.get("found"):
                    return SupportResponse(answer="No application matched that reference.",intent="application_status",trace_id="crewai-mock")
                answer=f"Application {application['record_id']} is {application['status']}; recorded amount INR {application['loan_amount_inr']:,}. " + ("This case should be routed for additional review." if application.get("escalate") else "No additional escalation is indicated by this score.")
                return SupportResponse(answer=answer,intent="application_status",application=application,escalated=bool(application.get("escalate")),trace_id="crewai-mock")
            else:
                from .rag import grounded_answer
                answer=grounded_answer(query)["answer"]; intent="policy_question"
            return SupportResponse(answer=answer,intent=intent,trace_id="crewai-mock")

    @tool("rag_lookup")
    def rag_lookup(query: str) -> str:
        """Retrieve grounded policy context for a member question."""
        COUNTERS["rag_calls"]+=1; return json.dumps(grounded_answer(query))
    @tool("check_loan_application_status")
    def lookup(record_id: str) -> str:
        """Look up a synthetic loan application by record ID."""
        COUNTERS["lookup_calls"]+=1; return json.dumps(check_loan_application_status(record_id))

    llm=MockLLM()
    retrieval=Agent(role="Retrieval Agent",goal="Retrieve only relevant policy context",backstory="Policy retrieval specialist",llm=llm,tools=[rag_lookup],allow_delegation=False,verbose=False)
    lookup_agent=Agent(role="Lookup Agent",goal="Check a single application reference",backstory="Authorized application lookup specialist",llm=llm,tools=[lookup],allow_delegation=False,verbose=False)
    composer=Agent(role="Response Composer",goal="Compose a concise safe member response",backstory="Support response writer",llm=llm,allow_delegation=False,verbose=False)
    if re.search(r"\bCRD-\d{4}\b",query,re.I):
        source_task=Task(description=f"Use the application status tool for this exact request: {query}",expected_output="Status lookup result.",agent=lookup_agent)
    else:
        source_task=Task(description=f"Retrieve policy context for this exact request: {query}",expected_output="Relevant policy passages.",agent=retrieval)
    compose_task=Task(description=f"Compose a concise, grounded response to this member request: {query}",expected_output="A safe answer grounded in the preceding tool result.",agent=composer,context=[source_task],
                       output_pydantic=SupportResponse)
    crew=Crew(agents=[retrieval,lookup_agent,composer],tasks=[source_task,compose_task],process=Process.sequential,verbose=False)
    draft=crew.kickoff()
    # Native crew has run; project contract is normalized/validated at boundary.
    if hasattr(draft,"pydantic") and draft.pydantic:
        return SupportResponse.model_validate(draft.pydantic.model_dump())
    return SupportResponse(answer=str(draft),intent="application_status" if "CRD-" in query else "policy_question",trace_id="crewai")
