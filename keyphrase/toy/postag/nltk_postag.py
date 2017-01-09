from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize

# Alternatively to setting the CLASSPATH add the jar and model via their path:
jar = '/Users/memray/Project/stanford/stanford-postagger/stanford-postagger.jar'
# model = '/Users/memray/Project/stanford/stanford-postagger/models/english-left3words-distsim.tagger'
model = '/Users/memray/Project/stanford/stanford-postagger/models/english-bidirectional-distsim.tagger'

pos_tagger = StanfordPOSTagger(model, jar)

# Add other jars from Stanford directory
stanford_dir = jar.rpartition('/')[0]
stanford_jars = find_jars_within_path(stanford_dir)
pos_tagger._stanford_jar = ':'.join(stanford_jars)

text = pos_tagger.tag(word_tokenize("What's the airspeed of an unladen swallow ?"))
print(text)
text = pos_tagger.tag('What is the airspeed of an unladen swallow ?'.split())
print(text)
