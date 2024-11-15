if __name__ == "__main__":
    actual = open("small_transcription.txt","r")
    expected = open("speech_true_text.txt","r")

    file_out = open("speech_dataset_tiny.tsv",'w')

    actual_lines = actual.readlines()
    expected_lines = expected.readlines()

    file_out.write("ID\tactual\texpected\n")

    for i in range(0, len(actual_lines)):
        file_out.write(str(i) + "\t" + actual_lines[i].strip() + "\t" + expected_lines[i].strip() + "\n")