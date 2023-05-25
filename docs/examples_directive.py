import os
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)

class ReadmeSectionsDirective(Directive):
    has_content = False
    required_arguments = 1


    def run(self):
        directory = self.arguments[0]
        sections = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower() == "readme.rst":
                    readme_path = os.path.join(root, file)
                    with open(readme_path, "r") as f:
                        content = f.read()
                    dir_name = os.path.basename(os.path.dirname(readme_path))
                    section = nodes.section(ids=[str(root)])
                    section_title = nodes.title(text=dir_name)
                    section.append(section_title)
                    section_body = nodes.paragraph(text=content)
                    section.append(section_body)

                    subsection = nodes.section(ids=[str("Test")])
                    subsection_title = nodes.title(text="Test", level=2)
                    subsection += subsection_title
                    subsection_content = nodes.paragraph(text="TEST TEST TEST")
                    subsection += subsection_content
                    section += subsection


                    sections.append(section)
        return sections

def setup(app):
    app.add_directive('readme_sections', ReadmeSectionsDirective)
