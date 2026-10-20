part_files = [
    "iSign-poses_v1.1_part_aa",
    "iSign-poses_v1.1_part_ab",
    "iSign-poses_v1.1_part_ac",
    "iSign-poses_v1.1_part_ad"
]

with open("iSign-poses_v1.1.zip", "wb") as output_file:
    for file_name in part_files:
        print(f"Merging {file_name}...")
        with open(file_name, "rb") as input_file:
            output_file.write(input_file.read())

print("Successfully merged into iSign-poses_v1.1.zip!")