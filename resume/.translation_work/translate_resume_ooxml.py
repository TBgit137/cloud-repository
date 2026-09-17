from __future__ import annotations

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path

from lxml import etree


ROOT = Path(r"D:\work\cloud-repository\resume")
SOURCE = ROOT / ".translation_work" / "source_cn.docx"
OUTPUT = ROOT / "个人简历_英文版.docx"
EXPECTED_SOURCE_HASH = "e1f55378e76d2a7fbc86165bc45434a3f75efd1fcfe0457ff9b13e8fb0a327b4"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_NS = "http://www.w3.org/XML/1998/namespace"
NS = {"w": W_NS}


# Each entry maps a body-paragraph index to replacements for selected run indexes.
# Run, paragraph, and style XML are never rebuilt; only existing w:t text changes.
REPLACEMENTS: dict[int, dict[int, str]] = {
    0: {0: "Resume"},
    2: {0: "Name: Xiaoyu Luo   Date of Birth: August 2004   Height: 175 cm  "},
    3: {0: "Contact: "},
    5: {0: "Education"},
    6: {0: "South China Normal University - Software Engineering - Bachelor's Degree"},
    7: {
        0: "The Hong Kong University of Science and Technology ",
        2: " Artificial Intelligence and Entrepreneurship (AIE) - ",
        3: "Full-time Postgraduate Student",
        7: "Present",
    },
    9: {
        0: "Academic Performance",
        1: " (South China Normal University)",
        2: ": ",
    },
    11: {
        0: "Major Courses: ",
        1: (
            "Python Programming, Computer Architecture, Object-Oriented Programming, Introduction to Software "
            "Engineering, Java Programming, Data Structures and Algorithms, Algorithm Design and Analysis, "
            "Operating Systems, Database Systems, Web Development, Linux, Computer Networks, Software System "
            "Design, Software Testing, AI Fundamentals, Network Security, Data Mining, Robotics, Software "
            "Design, Distributed Systems, Knowledge Representation, Natural Language Processing, Computational "
            "Intelligence."
        ),
        **{i: "" for i in range(2, 11)},
    },
    13: {0: "Project Experience"},
    15: {0: "Server Operations (Internship)"},
    16: {
        0: (
            "Completed a one-month internship at Guangzhou Wanglv Internet Technology Co., Ltd., managing "
            "servers, optimizing back-end algorithms, and writing bastion-host data migration scripts. Gained "
            "experience with Python/FastAPI back-end setup and operations, Nginx proxies, Linux, and scripting. "
            "Learned the software project lifecycle from requirements analysis and code review to deployment, "
            "while collaborating with colleagues."
        ),
        **{i: "" for i in range(1, 9)},
    },
    18: {
        0: "Online Collaborative Note-Taking Platform ",
        2: " Back-End Server and Database Setup and Operations",
    },
    19: {
        0: (
            "Earlier back ends used lightweight FastAPI, which offered limited support for business-layer and "
            "database interaction. This project adopted Java Spring Boot, progressing from Java Web fundamentals "
            "to enterprise framework development. Used presentation, business, and data layers to receive "
            "front-end data, process business logic, and perform CRUD operations on user data."
        ),
        **{i: "" for i in range(1, 10)},
    },
    21: {
        0: "Binary Fuzzing Agent (Internship) ",
        2: " Full Development",
    },
    22: {
        0: (
            "Completed a three-month internship at Beijing Yanling Wangwei Intelligent Technology Co., Ltd., "
            "developing a binary fuzzing agent. Learned taint analysis and fuzzing for binary executables, then "
            "focused on an AFL-based automated agent using LLMs for initial test-case generation, seed mutation, "
            "result analysis, and report generation."
        ),
        **{i: "" for i in range(1, 6)},
    },
    24: {
        0: "Airport Runway Scheduling Optimization Algorithm Design",
        3: " Capstone Project",
    },
    25: {
        0: (
            "Designed an ant-colony-based runway scheduling solution that minimizes total flight delay under "
            "constraints including runway count, minimum safety separation, and scheduled takeoff and landing "
            "times. In the front-end/back-end architecture, the front end collects and validates airport and "
            "flight parameters; the back end runs the optimization and returns the optimized sequence and "
            "scheduling recommendations for display."
        ),
        1: "",
        2: "",
    },
    27: {0: "Skills and Strengths", 1: ""},
    28: {
        0: "Computer Skills: ",
        1: (
            "Proficient in Python, Java, and C fundamentals; able to read and write basic assembly; experienced "
            "with databases and MySQL; trained in Linux operations and scripting; experienced in Web development "
            "with HTML, CSS, JavaScript, and Vue; proficient in FastAPI- and Spring-Boot-based back-end architectures."
        ),
        **{i: "" for i in range(2, 19)},
    },
    29: {
        0: "Teamwork: ",
        1: (
            "Strong communication skills and effective working relationships with team members; familiar with "
            "task allocation, version control, and collaborative division of work through project experience."
        ),
        **{i: "" for i in range(2, 5)},
    },
    30: {
        0: "Language Skills: ",
        1: "IELTS score of 7.0; able to read English texts without difficulty and communicate confidently in English.",
        2: "",
    },
    32: {0: "Self-Assessment", 1: ""},
    33: {
        0: (
            "Conscientious and responsible, with strong self-learning and problem-solving abilities and no habit "
            "of procrastination. Skilled at using online resources and a personal AI-assistant toolchain, with "
            "basic video, image, audio, and text editing skills. Value practical application of software "
            "engineering skills and hope to gain further hands-on development and industry experience."
        ),
        **{i: "" for i in range(1, 5)},
    },
    35: {
        0: (
            "My undergraduate studies covered many topics at an introductory level. Through internships, I hope "
            "to study project development more deeply, including its high-level logic and underlying principles "
            "in areas such as data processing and software development."
        ),
        1: "",
        2: "",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def set_run_text(run: etree._Element, value: str) -> None:
    text_nodes = run.xpath(".//w:t", namespaces=NS)
    if not text_nodes:
        if value:
            raise ValueError("Cannot place text into a run with no w:t node")
        return
    text_nodes[0].text = value
    if value.startswith(" ") or value.endswith(" "):
        text_nodes[0].set(f"{{{XML_NS}}}space", "preserve")
    else:
        text_nodes[0].attrib.pop(f"{{{XML_NS}}}space", None)
    for node in text_nodes[1:]:
        node.text = ""
        node.attrib.pop(f"{{{XML_NS}}}space", None)


def patch_document_xml(xml_bytes: bytes) -> bytes:
    parser = etree.XMLParser(remove_blank_text=False)
    root = etree.fromstring(xml_bytes, parser)
    paragraphs = root.xpath("/w:document/w:body/w:p", namespaces=NS)
    if len(paragraphs) != 37:
        raise ValueError(f"Expected 37 body paragraphs, found {len(paragraphs)}")
    for paragraph_index, run_map in REPLACEMENTS.items():
        runs = paragraphs[paragraph_index].xpath(".//w:r", namespaces=NS)
        for run_index, replacement in run_map.items():
            if run_index >= len(runs):
                raise IndexError(f"Paragraph {paragraph_index} has no run {run_index}")
            set_run_text(runs[run_index], replacement)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone="yes")


def main() -> None:
    if sha256(SOURCE) != EXPECTED_SOURCE_HASH:
        raise RuntimeError("Source hash changed; refusing to patch a different document")

    with tempfile.TemporaryDirectory(prefix="resume_translate_") as temp_dir:
        temp_output = Path(temp_dir) / OUTPUT.name
        with zipfile.ZipFile(SOURCE, "r") as src, zipfile.ZipFile(temp_output, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "word/document.xml":
                    data = patch_document_xml(data)
                dst.writestr(item, data)
        shutil.copyfile(temp_output, OUTPUT)

    print(OUTPUT)


if __name__ == "__main__":
    main()
