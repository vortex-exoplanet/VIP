"""
Update intro text in documentation using text in README.rst

"""


with open("README.rst") as f_r:
    readme = f_r.readlines()

start_write=False
stop_write=False
str_start = 'TL;DR'
str_stop = 'Mailing list'
with open("docs/source/trimmed_readme.rst", 'w') as f_i:
    for l, line in enumerate(readme):
        if start_write and not str_stop in line and not stop_write:
            f_i.write(line)
        elif str_start in line and not stop_write:
            f_i.write(line)
            start_write=True
        elif str_stop in line:
            stop_write=True
            l_about = l
        else:
            continue
            
with open("docs/source/about.rst", 'w') as f_i:
    for l, line in enumerate(readme):
        if l > l_about+1:
            line.replace('^', '-')
            f_i.write(line)
            
            

print("successfully converted README into documentation rst files.")
