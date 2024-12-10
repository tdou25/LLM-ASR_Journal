from openai import OpenAI
"""
Untested concept file for 1-shot prompting
"""
if __name__ == "__main__":
    keyfile = open("locked.txt", "r")
    key = keyfile.read()
    print(key)
    client = OpenAI(
        api_key=key
    )
    
    file = open("tinySet.csv", "r")
    fileOut = open("4o-test_prompt_output.csv","w")

    fileOut.write("RESPONSE,ACTUAL\n")

    Lines = file.readlines()
    
    num = 0

    for line in Lines:
        words = line.split(",")
        label = words[-1].strip()  # Remove newline character

        #0-shot approach
        full_message = "You are an corrective model for ASR correction. The following is a corrected ASR output:\
        input: \"So let it be with Caesar. The noble fruited \"\
        corrected output: So let it be with Caesar. The noble Brutus\
        \nPlease correct this line " + label

        full_message = full_message + words[0] + "\""
        
        print(num, full_message)

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": full_message,
                }
            ],
            model="gpt-4o",
        )
    
        content = chat_completion.choices[0].message.content

        fileOut.write(label + "," + content + "\n")

        num += 1
        