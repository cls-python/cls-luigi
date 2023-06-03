# -*- coding: utf-8 -*-
#
# Apache Software License 2.0
#
# Copyright (c) 2022-2023, Jan Bessai, Anne Meyer, Hadi Kutabi, Daniel Scholtyssek
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.fileutil import copy_asset
from sphinx.util.docutils import new_document
from docutils.core import publish_parts
from docutils.io import FileInput
from sphinx.util import logging

logger = logging.getLogger(__name__)

class ReadmeSectionsDirective(Directive):
    has_content = False
    required_arguments = 1
    final_argument_whitespace = True

    def run(self):
        directory = self.arguments[0]
        sections = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower() == "readme.rst":

                    readme_path = os.path.join(root, file)
                    dir_name = os.path.basename(os.path.dirname(readme_path))
                    logger.info(str(dir_name))
                    if dir_name == "getting_started":
                        break
                    with open(readme_path, "r") as f:
                        content = f.read()

                    doc = new_document(readme_path)
                    doc.settings = self.state.document.settings


                    build_path = self.state.document.settings.env.app.builder.outdir


                    # # Convert RST content to a list of strings
                    # rst_lines = content.splitlines()
                    # rst_string_list = StringList(rst_lines)

                    # for item in rst_string_list:
                    #     logger.info(str(item))

                    # # Parse the RST content
                    # self.state.nested_parse(rst_string_list, 0, doc)

                    # Read the content of the HTML file
                    with open(os.path.join(build_path,"examples", str(dir_name), str(file).replace(".rst", ".html")), 'r', encoding='utf-8') as file:
                        html_content = file.read()

                    # Create a raw HTML node and add it to the section
                    raw_node = nodes.raw('', html_content, format='html')
                    section_node = nodes.section(ids=[str(root)])
                    section_node.append(raw_node)

                    # section = nodes.section(ids=[str(root)])
                    # section_title = nodes.title(text=dir_name)
                    # section.append(section_title)
                    # section_body = nodes.paragraph("","",*doc.children)
                    # section.append(section_body)
                    sections.append(section_node)

        return sections


def setup(app):
    app.add_directive('readme_sections', ReadmeSectionsDirective)
