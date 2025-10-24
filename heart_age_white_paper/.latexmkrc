# Route all aux/build files to build/ directory
$out_dir = 'build';
$aux_dir = 'build';

# Ensure the directories exist
if (! -d $out_dir) { system("mkdir -p $out_dir"); }
if (! -d $aux_dir) { system("mkdir -p $aux_dir"); }

# Use pdflatex
$pdf_mode = 1;

# Clean-up list (so latexmk -C removes build artifacts)
@generated_exts = (@generated_exts, 'aux', 'bbl', 'blg', 'fdb_latexmk', 'fls', 'log', 'out', 'toc', 'lof', 'lot');
