# Project Ultron

![Ultron](Ultron.webp)

## What's project Ultron? 

Project Ultron is a _Question Answering Network_ (QANet for short) which uses [BERT](https://en.wikipedia.org/wiki/BERT_(language_model)) as a backend. For this particular projcet, I user [ktrain](https://github.com/amaiya/ktrain) which is a great interface for _Keras_. 

## How to run this project? 

Just clone the repository then run: 

```
pip install -r requirements.txt
```

And wait for the requirements to be installed. Remember in order to get it to work, you have to have _tensorflow_ installed on your system as well. This is tested on Linux, test it on macOS or Windows and give me feedbacks. 

_NOTE_: For the first run, it needs to download a few gigabytes of needed data for _BERT_. So be patient. 

## Code explanation

```python
import ktrain
from ktrain import text
``` 

These lines are obvious, I just imported what I needed. 

```python
INDEXDIR = '/tmp/index_file'; 
input_file = open('input_data.txt')
input_data = [line for line in input_file.readlines()]
``` 

In these lines, you can see an _index directory_ which will be made by the code after running. The next is the input file. It's not big enough but it's good enough to show the purpose of the code. 

In the third line, we just create a list from that input file. If you have more complex data (such as religious scriptures or really long books or data from google groups ... ), you obviously will need a much better preparation. 

```python
text.SimpleQA.initialize_index(INDEXDIR)
text.SimpleQA.index_from_list(input_data, INDEXDIR, commit_every=len(input_data), multisegment=True, procs=4, breakup_docs=True)
``` 

In these lines, we just make the brain. 

```python
qa = text.SimpleQA(INDEXDIR)
answers = qa.ask("Who are you?")
for answer in answers[:5]:
    print(answer)
```

Here, Ultron will answer you nicely. 

If you need a better explanation, just read [this notebook](https://github.com/amaiya/ktrain/blob/master/examples/text/question_answering_with_bert.ipynb). 