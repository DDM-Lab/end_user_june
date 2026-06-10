"""Prompt construction.

Per the Experiment Protocol (Study Specifics → LLM subjects), the LLM is
told upfront how the task works (one system message), then receives a
series of emails as conversational turns. Each turn it must reply with a
JSON object containing (a) the chosen action and (b) the
`relevant_strings_dict` — its self-reported saliency map.

The chat history is preserved across all 56 turns so the model "remembers"
what it has already seen, per the experiment design.
"""

from __future__ import annotations

from .types import EMAIL_RESPONSE_LIST, EmailContent


_EMAIL_RESPONSE_DICT_BLOCK = """email_response_dict = {
    "Archive": ["Move the email out of your inbox and store it for future reference without deleting it."],
    "Mark as Spam": ["Flag the email as unwanted or suspicious so similar messages are filtered in the future."],
    "Delete": ["Permanently remove the email from your inbox."],
    "Reply": ["Send a response directly back to the sender of the email."],
    "Download Attachments": ["Save any files included in the email onto your device."],
    "Click Link": ["Open a hyperlink in the email, typically directing you to an external webpage."]
}"""


SYSTEM_PROMPT = f"""In this study, we are interested in examining how LLMs read and respond to email messages. For this experiment, you will be presented with a series of 56 emails, one at a time, in the order they arrive in your inbox. After reading each email, perform the following tasks:

(1) Select the response out of six possible responses in email_response_list that is most appropriate for the email.
    (a) email_response_list = {EMAIL_RESPONSE_LIST!r}
    (b) Here is a dictionary that specifies what each action string represents:
{_EMAIL_RESPONSE_DICT_BLOCK}

(2) Output a dictionary called relevant_strings_dict, and fill it with keys that correspond to exact substrings within the email that were most important or relevant to determining your email response. For values within this dictionary, explain in a few concise words why each substring was important or relevant to determining your response.
    (a) You may indicate text within the body of the email as well as the email recipient, title, sender, etc.
    (b) Substrings should be copied verbatim from the email so they can be located in the original text.
    (c) Example — for email_message = ["Dear Lillian, Please respond within 24 hours to this survey so that we can resolve the power outage. Sincerely, Bob (CEO)"]:
        relevant_strings_dict = {{
            "Dear Lillian": ["A familiar tone"],
            "- Sincerely, Bob Smith (CEO)": ["An important person in the company"]
        }}

Respond to every email with a single JSON object — no prose, no markdown fences — exactly in this form:

{{
  "action": "<one item from email_response_list>",
  "relevant_strings_dict": {{"<exact substring from this email>": "<why it mattered>", ...}}
}}

Remember the entire conversation as you proceed. You may use what you have seen in earlier emails to inform later decisions, just as a human reading an inbox would."""


def build_email_user_message(trial_order: int, email: EmailContent) -> str:
    """One per-trial user message — the email itself plus a short prompt."""
    return (
        f"Email {trial_order} of 56.\n\n"
        f"From: {email.sender_name} <{email.sender_address}>\n"
        f"Subject: {email.subject}\n"
        f"--- BEGIN EMAIL BODY ---\n"
        f"{email.body_text.rstrip()}\n"
        f"--- END EMAIL BODY ---\n\n"
        f"Respond with the JSON object as instructed."
    )
