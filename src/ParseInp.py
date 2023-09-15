import re


class Parser:
    def __init__(self):
        self.ptn = r'(\d)\.(\d)$'

    def valid(self, i_str):
        ret = re.match(self.ptn, i_str)
        a, b = 0, 0
        if ret:
            a = ret.group(1)
            b = ret.group(2)
        return [bool(ret), int(a), int(b)]


if __name__ == "__main__":
    p = Parser()
    strA = input("Inp: ")
    print(p.valid(strA))
