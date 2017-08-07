# Workflow
# Change output dir to site
# Run the conversion
# Replace '![png](<file>_files/ with ![png](/img/ipynb/<file>_files/
# Copy the <file>_files to theme_dir/img/

# from: <file>
# to: <dir>
# jupyter nbconvert --to markdown ../nbs/<file>.ipynb --output-dir sources/<dir>/
# mv sources/<dir>/<file>_files theme_dir/img/ipynb/<file>_files
# In sources/<file>.md, replace '![png](<file>_files/ with ![png](/img/ipynb/<file>_files/
# - use sed?
#

# Example:
# jupyter nbconvert --to markdown ../nbs/getting_started.ipynb --output-dir sources/getting-started/
