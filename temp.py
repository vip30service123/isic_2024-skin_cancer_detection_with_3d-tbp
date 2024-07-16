class Nothing:
    haha:int = None

    def __init__(self):
        self.haha = "lala"

    def nothing(self):
        print(self.haha)

class Haha(Nothing):
    pass


class What(Nothing):
    pass


for cls in Nothing.__subclasses__():
    print(cls)
