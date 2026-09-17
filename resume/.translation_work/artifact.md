# English resume translation contract

## Reference

- Source: `D:\work\cloud-repository\resume\个人简历.docx`
- Working copy: `D:\work\cloud-repository\resume\.translation_work\source_cn.docx`
- SHA-256: `e1f55378e76d2a7fbc86165bc45434a3f75efd1fcfe0457ff9b13e8fb0a327b4`
- Source render: `.translation_work/source_render/page-1.png` and `page-2.png`
- Pages: 2
- Sections: 1

## Fidelity requirement

- Translate visible Chinese text into English only.
- Do not change page setup, section properties, paragraph properties, styles, tabs, fonts, sizes, colors, bold, italics, spacing, indentation, hyperlinks, headers, footers, or package relationships.
- Preserve all package parts byte-for-byte except `word/document.xml`.
- Within `word/document.xml`, change only text-node contents and `xml:space` where required for leading or trailing spaces.
- Preserve every paragraph, run, tab, hyperlink, and run-property node in its original position.
- Translation may naturally reflow because English text length differs, but no formatting adjustment is permitted.

## Content coverage

- Translate all non-empty body paragraphs, including the title, personal information, education, coursework, projects, skills, self-assessment, and closing statement.
- Keep names, phone numbers, email address, dates, acronyms, programming languages, product names, and GPA values unchanged except for English capitalization conventions such as Nginx, MySQL, Vue, and LLM.

## Delivery gate

- The source file must remain unchanged and retain the recorded SHA-256.
- The final package must differ only in `word/document.xml`.
- Render and inspect every final page for clipping, overlap, missing text, or broken tabs.
