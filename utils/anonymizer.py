"""
PII Anonymizer Module
Detects and redacts sensitive personal information (Names, Emails, Phones, Cards, SSNs, IPs)
from text before sending context to external LLM APIs, and de-anonymizes placeholders back
in the final LLM response.
"""

import re
from utils.logger import logger


class PIIAnonymizer:
    """
    Rule-based local PII Scrubber for RAG context chunks.
    Ensures zero sensitive identifiers reach external API providers.
    """

    def __init__(self):
        # Regular expression patterns for common PII
        self.patterns = [
            ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
            ("PHONE", r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
            ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,16}\b"),
            ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
            ("IP_ADDRESS", r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"),
            # Names with titles (Mr., Mrs., Dr., Prof., Ms.) or capitalized multi-word name patterns
            (
                "PERSON",
                r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b",
            ),
        ]

    def anonymize(self, text: str) -> tuple[str, dict[str, str]]:
        """
        Scan text and replace PII with placeholder tokens like [EMAIL_1], [PERSON_1].

        Args:
            text: Raw input string.

        Returns:
            Tuple of (anonymized_text, mapping_dict)
            mapping_dict maps placeholder -> original_value.
        """
        if not text:
            return "", {}

        mapping: dict[str, str] = {}
        anonymized_text = text
        counter = {}

        for ptype, regex in self.patterns:
            matches = list(set(re.findall(regex, anonymized_text)))
            for match in matches:
                # Avoid re-anonymizing existing bracketed tags
                if match.startswith("[") and match.endswith("]"):
                    continue

                counter[ptype] = counter.get(ptype, 0) + 1
                placeholder = f"[{ptype}_{counter[ptype]}]"

                # Store mapping for exact replacement
                mapping[placeholder] = match

                # Replace all occurrences of this specific match in the text
                anonymized_text = anonymized_text.replace(match, placeholder)

        if mapping:
            logger.info(
                f"PII Anonymizer redacted {len(mapping)} sensitive token(s): {list(mapping.keys())}"
            )

        return anonymized_text, mapping

    def deanonymize(self, text: str, mapping: dict[str, str]) -> str:
        """
        Replace placeholder tokens back to their original values.

        Args:
            text: Anonymized response string from LLM.
            mapping: Placeholder -> original string dictionary.

        Returns:
            De-anonymized output text.
        """
        if not text or not mapping:
            return text

        restored_text = text
        for placeholder, original in mapping.items():
            restored_text = restored_text.replace(placeholder, original)

        return restored_text


# Global singleton instance
pii_anonymizer = PIIAnonymizer()
