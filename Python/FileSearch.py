import os

class FileSearch:
    def __init__(self, path, file_extensions):
        self.folder_count = 0
        self.file_count = 0
        self.line_count = 0
        self.recurse(path, file_extensions)
        print("Total number of folders = ", self.folder_count)
        print("Total number of files = ", self.file_count)
        print("Total number of lines = ", self.line_count)

    def recurse(self, path, file_extensions):
        if not isinstance(path, str):
            return
        if not os.path.isdir(path):
            print("Path not found: ", path)
            return

        for each in os.listdir(path):
            if os.path.isfile(each):
                self.line_count += self.recurseFiles(each, file_extensions)
            else:
                folder_path = os.path.join(path, each)
                if os.path.isdir(folder_path):
                    self.recurseFolder(folder_path, file_extensions)

    def recurseFolder(self, path, file_extensions):
        if os.path.isdir(path):
            self.folder_count += 1
            for each in os.listdir(path):
                full_path = os.path.join(path, each)
                if os.path.isdir(each):
                    self.recurse(self, full_path)
                else:
                    self.processFile(full_path, file_extensions)

    def processFile(self, file, ends_with):
        if os.path.isfile(file):
            for string in ends_with:
                if file.endswith(string):
                    self.file_count += 1
                    self.line_count += self.countLines(file)
        else:
            self.recurseFolder(file, ends_with)

    def countLines(self, file):
        count = 0
        file = open(file)
        lines = file.readlines()
        for line in lines:
            file.readline()
            count += 1
        file.close()
        return count

if __name__ == "__main__":
    dir = os.getcwd()
    dir = "C:\DEV\PhotoResearch\Test"
    dir = "C:\\DEV\\PhotoResearch\\MetaDeveloper1.05\\19xx"
    file_ext = [".bat", ".h", ".cpp"]
    fs = FileSearch(dir, file_ext)
