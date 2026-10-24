from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str = 'neil'#default value is neil
    age:Optional[int] = None
    email:EmailStr
    cgpa:float = Field(gt=0,lt=10,default=5,description="A decimal value respresent grade")
    #range constraints
new_student = {'age':32,'email':'anything123@gamil.com','cgpa':9}

student = Student(**new_student)

student_dict = dict(student)
#same with json

print(student)