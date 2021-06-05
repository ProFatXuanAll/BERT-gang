r"""Fine-tune teacher and student models.

Usage:
    import fine_tune

    teacher_model = fine_tune.model.TeacherAlbert(...)
    teacher_model = fine_tune.model.TeacherBert(...)

    student_model = fine_tune.model.StudentAlbert(...)
    student_model = fine_tune.model.StudentBert(...)
"""

# built-in modules

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Union

# my own modules

from fine_tune.model._student_albert import StudentAlbert
from fine_tune.model._student_bert import StudentBert
from fine_tune.model._teacher_albert import TeacherAlbert
from fine_tune.model._teacher_bert import TeacherBert
from fine_tune.model.logits_bank import Logitsbank
from fine_tune.model._gate import HighwayGate, LinearGate

# Define types for type annotation.

Model = Union[
    StudentAlbert,
    StudentBert,
    TeacherAlbert,
    TeacherBert
]

StudentModel = Union[
    StudentAlbert,
    StudentBert,
]

TeacherModel = Union[
    TeacherAlbert,
    TeacherBert,
]
