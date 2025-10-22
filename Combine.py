import os
import docx
import textract

#Reads a file as input and returns the content
#Looks for txt, docx and doc files, if the extension is find, returns the content
#If it is not find, returns nothing
def readfile(filepath):
    if filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    elif filepath.endswith('.docx'):
        doc = docx.Document(filepath)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif filepath.endswith('.doc'):
        try:
            return textract.process(filepath).decode('utf-8')
        except Exception as e:
            return f"ERROR: Can't find file ({filepath}): {str(e)}"
    else:
        return None

#Writes the content to "combined_content.txt"
def write_to_file(content, outputfile="combined_content.txt"):
    with open(outputfile, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"\nAll content is saved in: {outputfile}")

#This function get every file that is eligible with the features above from the map that is defined as map_path and extracts the content
def read_and_combine(map_path, outputfile="combined_content.txt"):
    combined_content_v = []

    #For every file in the map, the content is read and merged in combined_content_v
    #If there is no content, the file is skipped and the next file is taken
    for filename in os.listdir(map_path):
        filepath = os.path.join(map_path, filename)
        if os.path.isfile(filepath):
            content = readfile(filepath)
            if content is not None:
                combined_content_v.append(f"\n\n=== File: {filename} ===\n")
                combined_content_v.append(content)
            else:
                print(f"File skipped: {filename}")
    #After all, everything is merged and written to the outputfiel
    merge = "\n".join(combined_content_v)
    write_to_file(merge, outputfile)

map_path = 'C:/Users/harpr/Documents/Master/Thesis/Data/25_series/'
outputfile = "combined_content.txt"
read_and_combine(map_path, outputfile)