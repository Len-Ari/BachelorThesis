input_file = "tmp.txt"
output_file = "requirements.txt"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        if "==" in line:  # Only process lines with version specifications
            package, version = line.split("==", maxsplit=1)
            # Write package and version without the build tag
            f_out.write(f"{package}=={version.split('=')[0]}\n")
