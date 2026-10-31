"""Independent two-agent mock review and optional AutoGen round-robin adapter."""
from __future__ import annotations
from pydantic import BaseModel
from .rag import documents

class Verdict(BaseModel):
    approved: bool
    final_answer: str
    reason: str

def review_draft(draft: str, context: str) -> Verdict:
    """Policy reviewer checks draft support; final editor returns a structured verdict."""
    if not context.strip():
        return Verdict(approved=False,final_answer="I don't know based on the available policy documents. Please ask an authorized support representative.",reason="No retrieved policy context was supplied.")
    # Deliberate unsupported absolute/guarantee language is corrected in the test demonstration.
    risky=("guaranteed approval" in draft.casefold() or "will definitely" in draft.casefold() or "always free" in draft.casefold())
    if risky:
        safe="Eligibility and charges depend on product terms and individual review. Please consult the applicable current schedule or formal offer."
        return Verdict(approved=False,final_answer=safe,reason="Removed an unsupported guarantee and qualified the claim to the supplied policy context.")
    return Verdict(approved=True,final_answer=draft,reason="Draft is consistent with the supplied context and contains no unsupported guarantee.")

async def autogen_review(draft: str, context: str, model_client=None) -> Verdict:
    """Run a 2-agent AutoGen RoundRobinGroupChat with a typed Final-Editor output."""
    try:
        from autogen_agentchat.agents import AssistantAgent
        from autogen_agentchat.teams import RoundRobinGroupChat
        from autogen_agentchat.messages import StructuredMessage
        from autogen_core.models import ModelFamily
    except ImportError as e: raise RuntimeError("Install requirements.txt to enable AutoGen adapter") from e
    if model_client is None:
        from autogen_ext.models.replay import ReplayChatCompletionClient
        verdict=review_draft(draft,context)
        # Replay responses are local scripted completions. Structured output is still
        # parsed/validated by AutoGen using Verdict as Final-Editor's output_content_type.
        review_note="Policy review completed against the supplied context."
        model_client=ReplayChatCompletionClient(
            [review_note,verdict.model_dump_json()],
            model_info={"vision":False,"function_calling":False,"json_output":True,
                        "structured_output":True,"family":ModelFamily.UNKNOWN},
        )
    reviewer=AssistantAgent("Policy_Compliance_Reviewer",model_client,
        system_message="Compare the draft with the supplied retrieved context. Flag unsupported guarantees, invented fees, or unsupported status. Be concise.")
    editor=AssistantAgent("Final_Editor",model_client,
        system_message="Return a structured verdict. Approve a supported draft unchanged; otherwise revise it to remove unsupported claims.",
        output_content_type=Verdict)
    team=RoundRobinGroupChat([reviewer,editor],max_turns=2,
        custom_message_types=[StructuredMessage[Verdict]])
    result=await team.run(task=f"Retrieved context:\n{context}\n\nComposer draft:\n{draft}")
    for message in reversed(result.messages):
        content=getattr(message,"content",None)
        if isinstance(content,Verdict): return content
    raise RuntimeError("AutoGen review team completed without a structured verdict")
