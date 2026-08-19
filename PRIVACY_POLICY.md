# 🔒 Privacy Policy & Code Usage Terms

*Last Updated: August 2026*

This document outlines the **Code Usage & Intellectual Property Terms** for this repository as well as the **Data Privacy & Security Commitments** enforced by the Ask My Documents RAG application.

---

## 1. 📜 Code Accessibility & Intellectual Property Terms

### Open Code vs. Open Source
- **Public Visibility ("Open Code")**: The source code for this project is publicly visible on GitHub to allow transparent inspection, technical evaluation, and security auditing.
- **NOT Open Source**: This software is **NOT open source** under the Open Source Definition (OSI) or any Free/Libre Open Source Software (FLOSS) license.
- **Consent Requirement**: Public access to this repository does **NOT** constitute permission or a license to copy, modify, distribute, execute, host, deploy, or commercially exploit the codebase.
- **Strict Prohibition**: You may **NOT** use, copy, modify, re-license, redistribute, host, or deploy this code or any part of its architecture, algorithms, or user interface without explicit prior written consent from the author (**Sudhanshu Batra**).

For usage rights, commercial inquiries, or permissions, please contact [Sudhanshu Batra on GitHub](https://github.com/Sudhanshub27).

---

## 2. 🛡️ Application Data Privacy & Security Commitments

The Ask My Documents application is engineered with a strict privacy-first architecture to protect all uploaded documents and user queries:

### A. Zero-Training AI Model Compliance
- **No LLM Training**: Uploaded document content, extracted embeddings, and query interactions are **NEVER** used to train, fine-tune, or improve public or proprietary AI models.
- **Default Inference Provider**: The application uses Groq API as its primary inference engine. Groq's official terms of service contractually prohibit data retention and model training on API inputs and outputs.
- **Bring Your Own Key (BYOK)**: When users provide custom API keys (Groq, OpenAI, Anthropic, DeepSeek, Gemini, OpenRouter), API requests are transmitted directly under zero-retention developer parameters.

### B. Local Client-Side PII Scrubbing
- Before any document chunk or query payload is sent to an external inference endpoint, a local PII regex anonymization scrubber (`utils/anonymizer.py`) sanitizes personal information:
  - Personal Names & Contact References
  - Email Addresses (`[EMAIL_REDACTED]`)
  - Phone Numbers (`[PHONE_REDACTED]`)
  - IPv4 and IPv6 Addresses (`[IP_REDACTED]`)

### C. Multi-Tenant Session Isolation
- **No Account Creation**: Users do not need to register an account, provide an email address, or create login credentials.
- **HttpOnly Cookie Isolation**: Sessions are automatically scoped via an anonymous `rag_tenant_id` cookie.
- **Physical Vector Scoping**: Vector embeddings in ChromaDB (`tenant_<tenant_id>`), sparse BM25 indices, and document summary caches are strictly isolated per tenant ID. Cross-tenant data leakage is structurally prevented.

### D. User Data Control & Instant Wipe
- Users maintain 100% control over their uploaded data.
- Executing a tenant wipe (via the application UI or `/api/tenant/{tenant_id}`) immediately and permanently drops the tenant vector collection, clears keyword indices, and deletes associated summary cache files from disk.

---

## 3. 🍪 Cookie & Web Storage Policy

- **Session Scoping Cookies**: The application uses a single `HttpOnly`, `SameSite=Lax` cookie (`rag_tenant_id`) exclusively for multi-tenant database routing.
- **Zero Third-Party Advertising Cookies**: No tracking, advertising, or cross-site behavioral telemetry cookies are used.

---

## 4. 📞 Contact & License Inquiries

For questions regarding data privacy or to request explicit permission/consent for code usage:

- **Author**: Sudhanshu Batra
- **GitHub**: [@Sudhanshub27](https://github.com/Sudhanshub27)
- **Repository**: [https://github.com/Sudhanshub27/rag-system](https://github.com/Sudhanshub27/rag-system)
