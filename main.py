import streamlit as st
from st_on_hover_tabs import on_hover_tabs
from typing import List, Dict
from langchain_groq import ChatGroq
from langchain_classic.chains import LLMChain, ConversationChain
from langchain_classic.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory
import json
from docx import Document
import os, re
import subprocess
import tempfile
from streamlit_lottie import st_lottie

# ── Page config ────────────────────────────────────────────────────────────────
hide_menu_style = """
<style>
MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)

try:
    with open("style.css") as f:
        st.markdown('<style>' + f.read() + '</style>', unsafe_allow_html=True)
except Exception:
    pass

dark_purple_theme = """
<style>
    :root {
        --primary-color: #7B2CBF;
        --background-color: black;
        --secondary-bg: #2D2D2D;
        --text-color: #FFFFFF;
        --accent-color: #9D4EDD;
    }
    .stApp { background-color: var(--background-color); color: var(--text-color); }
    h1, h2, h3 { color: var(--accent-color) !important; }
    .stButton>button {
        background-color: var(--primary-color); color: white;
        border: none; border-radius: 4px; transition: all 0.3s ease;
    }
    .stButton>button:hover { background-color: var(--accent-color); transform: translateY(-2px); }
    .stTextArea>div>div>textarea {
        background-color: var(--secondary-bg); color: var(--text-color);
        border: 1px solid var(--primary-color);
    }
    .stFileUploader { background-color: var(--secondary-bg); border: 1px dashed var(--primary-color); border-radius: 4px; }
    .stProgress > div > div > div > div { background-color: var(--primary-color); }
    .stChatMessage { background-color: var(--secondary-bg); border-radius: 8px; padding: 10px; margin: 5px 0; }
</style>
"""
st.markdown(dark_purple_theme, unsafe_allow_html=True)


# ── API Key gate ───────────────────────────────────────────────────────────────
def api_key_sidebar() -> str:
    """Render API key input in sidebar; key is never persisted beyond session RAM."""
    with st.sidebar:
        st.markdown("### 🔑 Groq API Key")
        key = st.text_input(
            "Enter your Groq API key",
            type="password",
            placeholder="gsk_...",
            help="Your key is never stored — it lives only in this browser session.",
        )
        if key:
            st.success("Key loaded ✓", icon="✅")
        else:
            st.warning("Paste your Groq API key to get started.")
    return key


# ── Lottie / About ────────────────────────────────────────────────────────────
def load_lottie_urls():
    return {
        "shield":   "https://assets5.lottiefiles.com/packages/lf20_yom6uvgj.json",
        "analysis": "https://assets8.lottiefiles.com/packages/lf20_qmfs6c3i.json",
        "chat":     "https://assets8.lottiefiles.com/packages/lf20_2LdLki.json",
        "security": "https://assets8.lottiefiles.com/packages/lf20_oyi9a28g.json",
    }


def about_section():
    animations = load_lottie_urls()
    st.title("GAMKERS Security Analysis Suite")
    st.write("---")

    with st.container():
        l, r = st.columns(2)
        with l:
            st.header("Advanced Security Analysis")
            st.write(
                "Our AI-powered security suite provides deep code inspection and vulnerability "
                "detection using state-of-the-art machine learning models."
            )
            st.button("🚀 Try Analysis Now", key="try_analysis")
        with r:
            st_lottie(animations["analysis"], height=300, key="analysis_anim")

    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st_lottie(animations["chat"], height=300, key="chat_anim")
        with r:
            st.header("Real-time Expert Consultation: GAMKERSGPT")
            st.write("Get instant security advice — vulnerabilities, best practices, threat detection.")
            st.button("💬 Start Chat", key="start_chat")

    st.write("---")
    with st.container():
        l, r = st.columns(2)
        with l:
            st.header("Comprehensive Security Suite")
            c1, c2 = st.columns(2)
            with c1:
                st.write("🔍 Code Obfuscation\n\n🌐 Network Analysis\n\n📂 File Operations")
            with c2:
                st.write("⚠️ Payload Detection\n\n🔐 API Security\n\n🛡️ Anti-Analysis")
        with r:
            st_lottie(animations["security"], height=300, key="security_anim")

    st.write("---")
    with st.container():
        st.header("Security Insights")
        c1, c2, c3 = st.columns(3)
        c1.metric("Languages Supported", "7+")
        c2.metric("Security Rules", "1000+")
        c3.metric("Analysis Speed", "<2 min")
    st.write("---")


# ── Core app class ─────────────────────────────────────────────────────────────
class SecurityAnalysisApp:
    def __init__(self, groq_api_key: str):
        self.chat_model = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="qwen/qwen3-32b",
            temperature=0.7,
            max_tokens=None,
        )

        self.analysis_template = PromptTemplate(
            input_variables=["code_chunk"],
            template="""Analyze the following code for malicious indicators and security concerns.

Code to analyze:
{code_chunk}

Return ONLY this JSON structure with no additional text:
{{
    "summary": ["Brief overview of key findings and potential security implications"],
    "sections": {{
        "code_obfuscation_techniques": {{
            "findings": [],
            "description": "Brief explanation of identified obfuscation techniques"
        }},
        "suspicious_api_calls": {{
            "findings": [],
            "description": "Overview of concerning API calls and their potential security impact"
        }},
        "anti_analysis_mechanisms": {{
            "findings": [],
            "description": "Summary of detected anti-analysis features"
        }},
        "network_communication_patterns": {{
            "findings": [],
            "description": "Analysis of network-related code patterns"
        }},
        "file_system_operations": {{
            "findings": [],
            "description": "Evaluation of file system interactions and associated risks"
        }},
        "potential_payload_analysis": {{
            "findings": [],
            "description": "Assessment of potential malicious payloads"
        }}
    }}
}}

Use "None identified" in findings array if no indicators are found.""",
        )

        self.binary_analysis_template = PromptTemplate(
            input_variables=["strings_chunk"],
            template="""Analyze the following strings extracted from a binary file for malicious indicators.

Extracted strings:
{strings_chunk}

Return ONLY this JSON structure with no additional text:
{{
    "summary": ["Brief overview of key findings and potential security implications"],
    "sections": {{
        "suspicious_strings": {{
            "findings": [],
            "description": "Identified suspicious strings and their implications"
        }},
        "command_and_control_indicators": {{
            "findings": [],
            "description": "Potential C2 indicators like URLs, IPs, or domain patterns"
        }},
        "anti_analysis_indicators": {{
            "findings": [],
            "description": "Strings suggesting anti-analysis capabilities"
        }},
        "network_related_strings": {{
            "findings": [],
            "description": "Network-related strings and security concerns"
        }},
        "file_system_indicators": {{
            "findings": [],
            "description": "File system related strings and associated risks"
        }},
        "potential_malware_functionality": {{
            "findings": [],
            "description": "Strings indicating malicious functionality"
        }}
    }}
}}

Use "None identified" in findings array if no indicators are found.""",
        )

        self.analysis_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.analysis_template,
            verbose=True,
        )
        self.binary_analysis_chain = LLMChain(
            llm=self.chat_model,
            prompt=self.binary_analysis_template,
            verbose=True,
        )
        self.chat_memory = ConversationBufferMemory()
        self.conversation = ConversationChain(
            llm=self.chat_model,
            memory=self.chat_memory,
            verbose=True,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────
    def clean_json_response(self, response: str) -> str:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != 0:
            response = response[start:end]
        return response.replace('```json', '').replace('```', '').strip()

    def analyze_chunk(self, chunk: str, is_binary: bool = False) -> Dict:
        try:
            if is_binary:
                response = self.binary_analysis_chain.predict(strings_chunk=chunk)
            else:
                response = self.analysis_chain.predict(code_chunk=chunk)
            cleaned = self.clean_json_response(response)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as je:
                return self._create_error_analysis("JSON parsing failed", str(je), is_binary)
        except Exception as e:
            return self._create_error_analysis("Analysis failed", str(e), is_binary)

    def _create_error_analysis(self, error_type: str, details: str, is_binary: bool = False) -> Dict:
        sections = (
            ["suspicious_strings", "command_and_control_indicators", "anti_analysis_indicators",
             "network_related_strings", "file_system_indicators", "potential_malware_functionality"]
            if is_binary else
            ["code_obfuscation_techniques", "suspicious_api_calls", "anti_analysis_mechanisms",
             "network_communication_patterns", "file_system_operations", "potential_payload_analysis"]
        )
        return {
            "error": f"{error_type}: {details}",
            "summary": [f"Analysis failed — {error_type}"],
            "sections": {s: {"findings": ["Analysis failed"], "description": "Technical error"} for s in sections},
        }

    def split_code_in_chunks(self, content: str, chunk_size: int = 12800) -> List[str]:
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    def analyze_code(self, code_content: str) -> Dict:
        chunks = self.split_code_in_chunks(code_content)
        analyses, progress_bar, status_text = [], st.progress(0), st.empty()
        for i, chunk in enumerate(chunks, 1):
            status_text.text(f"Analyzing chunk {i}/{len(chunks)}...")
            analyses.append(self.analyze_chunk(chunk))
            progress_bar.progress(i / len(chunks))
        status_text.text("Analysis complete!")
        progress_bar.empty()
        return self.combine_analyses(analyses)

    def extract_strings_from_binary(self, binary_data: bytes, min_length: int = 10) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.exe') as tmp:
            tmp.write(binary_data)
            tmp_path = tmp.name
        try:
            try:
                result = subprocess.run(
                    ['strings', '-n', str(min_length), tmp_path],
                    capture_output=True, text=True, check=True,
                )
                extracted = result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                extracted = self._extract_strings_manually(binary_data, min_length)
            filtered = "\n".join(
                line for line in extracted.splitlines() if re.search(r'[a-zA-Z0-9_]', line)
            )
            return " ".join(filtered.splitlines())
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _extract_strings_manually(self, binary_data: bytes, min_length: int = 4) -> str:
        strings, current = [], ""
        for byte in binary_data:
            if 32 <= byte <= 126:
                current += chr(byte)
            else:
                if len(current) >= min_length:
                    strings.append(current)
                current = ""
        if len(current) >= min_length:
            strings.append(current)
        return "\n".join(strings)

    def analyze_binary(self, binary_data: bytes) -> Dict:
        strings_content = self.extract_strings_from_binary(binary_data)
        chunks = self.split_code_in_chunks(strings_content)
        analyses, progress_bar, status_text = [], st.progress(0), st.empty()
        for i, chunk in enumerate(chunks, 1):
            status_text.text(f"Analyzing binary chunk {i}/{len(chunks)}...")
            analyses.append(self.analyze_chunk(chunk, is_binary=True))
            progress_bar.progress(i / len(chunks))
        status_text.text("Binary analysis complete!")
        progress_bar.empty()
        return self.combine_analyses(analyses)

    def combine_analyses(self, analyses: List[Dict]) -> Dict:
        if not analyses:
            return {"summary": ["No analysis results available"], "sections": {}, "errors": ["No analyses performed"]}
        combined = {"summary": set(), "sections": {}, "errors": []}
        if "sections" in analyses[0]:
            for s in analyses[0]["sections"]:
                combined["sections"][s] = {"findings": set(), "description": ""}
        for analysis in analyses:
            if "error" in analysis:
                combined["errors"].append(analysis["error"])
            combined["summary"].update(analysis.get("summary", []))
            for section, content in analysis.get("sections", {}).items():
                if section not in combined["sections"]:
                    combined["sections"][section] = {"findings": set(), "description": ""}
                combined["sections"][section]["findings"].update(content.get("findings", []))
                if content.get("description") and not combined["sections"][section]["description"]:
                    combined["sections"][section]["description"] = content["description"]
        result = {"summary": list(combined["summary"]), "sections": {}, "errors": combined["errors"]}
        for section, content in combined["sections"].items():
            findings = list(content["findings"])
            if len(findings) > 1 and "Analysis failed" in findings:
                findings.remove("Analysis failed")
            result["sections"][section] = {
                "findings": findings,
                "description": content["description"] or "No significant findings in this category",
            }
        return result

    def create_analysis_report(self, analysis_results: Dict, title: str = "Security Analysis Report") -> str:
        doc = Document()
        doc.add_heading(title, 0)
        doc.add_heading("Executive Summary", level=1)
        for point in analysis_results.get("summary", ["No summary available"]):
            doc.add_paragraph(point, style='Body Text')
        for section_name, content in analysis_results.get("sections", {}).items():
            doc.add_heading(section_name.replace('_', ' ').title(), level=1)
            if content.get("description"):
                doc.add_paragraph(content["description"], style='Body Text')
            if content.get("findings"):
                doc.add_heading("Findings:", level=2)
                for finding in content["findings"]:
                    if finding != "None identified":
                        doc.add_paragraph(f"• {finding}", style='List Bullet')
                    else:
                        doc.add_paragraph("No specific issues identified.", style='Body Text')
        if analysis_results.get("errors"):
            doc.add_heading("Analysis Errors", level=1)
            for error in analysis_results["errors"]:
                doc.add_paragraph(f"• {error}", style='List Bullet')
        filename = f"security_analysis_report_{os.getpid()}.docx"
        doc.save(filename)
        return filename

    def get_chat_response(self, user_input: str) -> str:
        return self.conversation.predict(input=user_input + " Response should be short and crisp.")


# ── Display helpers ────────────────────────────────────────────────────────────
def display_analysis_results(analysis: Dict):
    st.header("Executive Summary")
    for point in analysis.get("summary", ["No summary available"]):
        st.write(point)
    st.divider()
    if analysis.get("errors"):
        st.error("Analysis Errors")
        for error in analysis["errors"]:
            st.write(f"• {error}")
        st.divider()
    for section_name, content in analysis.get("sections", {}).items():
        st.subheader(section_name.replace('_', ' ').title())
        if content.get("description"):
            st.write(content["description"])
        if content.get("findings"):
            st.write("Findings:")
            for finding in content["findings"]:
                st.write(f"• {finding}")
        st.divider()


def download_report_button(report_filename: str, label: str = "📥 Download Analysis Report"):
    with open(report_filename, "rb") as f:
        st.download_button(
            label=label,
            data=f,
            file_name=report_filename,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    os.remove(report_filename)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # 1. Collect API key — never persisted to disk or env vars
    groq_api_key = api_key_sidebar()

    # 2. Tab navigation (rendered below the key widget in the sidebar)
    with st.sidebar:
        st.write("---")
        tabs = on_hover_tabs(
            tabName=['Code Analyzer', 'GAMKERSGPT', 'About'],
            iconName=['code', 'chat', 'info'],
            styles={
                'navtab': {
                    'background-color': 'black', 'color': '#9D4EDD',
                    'font-size': '16px', 'transition': '.3s',
                    'white-space': 'nowrap', 'text-transform': 'uppercase',
                },
                'tabOptionsStyle': {':hover': {'color': '#1A1A1A', 'background-color': 'black'}},
                'iconStyle': {'position': 'fixed', 'left': '7.5px', 'text-align': 'left'},
                'tabStyle': {'list-style-type': 'none', 'margin-bottom': '30px', 'padding-left': '30px'},
            },
        )

    # About page needs no API key
    if tabs == 'About':
        about_section()
        return

    # Gate all functional tabs behind a valid key
    if not groq_api_key:
        st.info("👈 Enter your Groq API key in the sidebar to get started.")
        return

    # Build (or reuse) the app instance; rebuild automatically if the key changes
    if "app" not in st.session_state or st.session_state.get("_api_key") != groq_api_key:
        st.session_state.app = SecurityAnalysisApp(groq_api_key)
        st.session_state.messages = []
        st.session_state._api_key = groq_api_key

    # ── Code Analyzer ─────────────────────────────────────────────────────────
    if tabs == 'Code Analyzer':
        st.title("GAMKERS Security Analyzer")
        tab1, tab2, tab3 = st.tabs(["📝 Paste Code", "📁 Upload Source File", "💾 Upload Binary"])

        with tab1:
            code_input = st.text_area("Paste your code here:", height=300)
            if st.button("🔍 Analyze Code", key="analyze_pasted") and code_input:
                with st.spinner("🔄 Analyzing code..."):
                    results = st.session_state.app.analyze_code(code_input)
                    display_analysis_results(results)
                    download_report_button(st.session_state.app.create_analysis_report(results))

        with tab2:
            uploaded_file = st.file_uploader(
                "Choose a source code file",
                type=['py', 'js', 'java', 'cpp', 'cs', 'php', 'rb'],
                key="code_file",
            )
            if st.button("🔍 Analyze Source File", key="analyze_source") and uploaded_file:
                with st.spinner("🔄 Analyzing source file..."):
                    results = st.session_state.app.analyze_code(uploaded_file.read().decode())
                    display_analysis_results(results)
                    download_report_button(st.session_state.app.create_analysis_report(results))

        with tab3:
            st.write("Upload a binary file (.exe) for security analysis")
            uploaded_binary = st.file_uploader("Choose a binary file", type=['exe'], key="binary_file")
            if uploaded_binary:
                st.info("Binary analysis extracts strings from the executable and analyzes them for security indicators.")
            if st.button("🔍 Analyze Binary", key="analyze_binary") and uploaded_binary:
                with st.spinner("🔄 Extracting strings and analyzing binary..."):
                    binary_data = uploaded_binary.read()
                    with st.expander("Extracted Strings Preview"):
                        extracted = st.session_state.app.extract_strings_from_binary(binary_data)
                        st.text_area(
                            "Strings from binary",
                            value=extracted[:10000] + ("\n\n[Truncated...]" if len(extracted) > 10000 else ""),
                            height=300, disabled=True,
                        )
                    results = st.session_state.app.analyze_binary(binary_data)
                    st.subheader("Binary Analysis Results")
                    display_analysis_results(results)
                    download_report_button(
                        st.session_state.app.create_analysis_report(
                            results, title=f"Binary Security Analysis Report - {uploaded_binary.name}"
                        ),
                        label="📥 Download Binary Analysis Report",
                    )

    # ── GAMKERSGPT ────────────────────────────────────────────────────────────
    elif tabs == 'GAMKERSGPT':
        st.title("GAMKERSGPT - Security Assistant")
        st.markdown(
            """
            <style>
            [data-testid="stChatMessage"] { background-color:#000 !important; color:#fff !important; padding:10px; border-radius:8px; }
            [data-testid="stChatInput"]   { background-color:#000 !important; color:#fff !important; border:1px solid #333; border-radius:8px; }
            [data-testid="stBottom"]      { background-color:#000 !important; padding:10px !important; border-radius:8px !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### Welcome to GAMKERSGPT!")
        st.write("Your cybersecurity assistant — ask about vulnerabilities, best practices, threat analysis, and more.")
        st.info("Tip: Be as specific as possible for the most relevant answers.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your cyber security question..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    response = st.session_state.app.get_chat_response(prompt)
                    response = response.split("</think>")[-1].strip()
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
