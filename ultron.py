import ktrain
from ktrain import text 

INDEXDIR = '/tmp/index_file'; 
input_file = open('input_data.txt')
input_data = [line for line in input_file.readlines()]

text.SimpleQA.initialize_index(INDEXDIR)
text.SimpleQA.index_from_list(input_data, INDEXDIR, commit_every=len(input_data), multisegment=True, procs=4, breakup_docs=True)