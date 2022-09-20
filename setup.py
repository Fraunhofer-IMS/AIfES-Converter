"""
Copyright (C) 2022  Fraunhofer Institute for Microelectronic Circuits and Systems.
All rights reserved.
AIfES-Converter is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="AIfES-Converter",
    version="1.0.0",

    description="This is a convert tool to create AIfES models for direct use in the Arduino IDE or other IDEs",

    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://aifes.ai/",
    author="Fraunhofer IMS",
    author_email="aifes@ims.fraunhofer.de",

    classifiers=[
        "Development Status :: 5 - Production/Stable",

        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",

        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Embedded Systems",

        "License :: OSI Approved :: GNU Affero General Public License v3",

        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],

    package_dir={"": "src"},
    packages=find_packages(where="src"),

    python_requires=">=3.7, <3.10",

    install_requires=[
                      "numpy>=1.19",
                      "packaging"
                      ],

    extras_require={
        "tensorflow": ["tensorflow>=2.4"],
        "pytorch": ["torch>=1.8"],
        "both": ["tensorflow>=2.4", "torch>=1.8"]
    },

    include_package_data=True,
    # data_files=[('./src/aifes/aifes_code_generator/templates/aifes', ['src/aifes/aifes_code_generator/templates/aifes/aifes_f32_fnn.h', 'src/aifes/aifes_code_generator/templates/aifes/aifes_f32_weights.h', 'src/aifes/aifes_code_generator/templates/aifes/aifes_q7_fnn.h', 'src/aifes/aifes_code_generator/templates/aifes/aifes_q7_weights.h']),
    #             ('./src/aifes/aifes_code_generator/templates/aifes_express',['src/aifes/aifes_code_generator/templates/aifes_express/aifes_e_f32_fnn.h', 'src/aifes/aifes_code_generator/templates/aifes_express/aifes_e_f32_weights.h','src/aifes/aifes_code_generator/templates/aifes_express/aifes_e_q7_fnn.h', 'src/aifes/aifes_code_generator/templates/aifes_express/aifes_e_q7_weights.h'])],

    project_urls={
        "Source": "https://github.com/Fraunhofer-IMS/AIfES-Converter/",
        "Bug Reports": "https://github.com/Fraunhofer-IMS/AIfES-Converter/issues",
        "Documentation": "https://fraunhofer-ims.github.io/AIfES-Converter/#index"
    },
)