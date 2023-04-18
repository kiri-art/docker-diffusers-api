class Status:
    def __init__(self):
        self.type = "init"
        self.progress = 0.0

    def update(self, type, progress):
        self.type = type
        self.progress = progress

    def get(self):
        return {"type": self.type, "progress": self.progress}


status = Status()
