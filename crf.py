def load_conll03_data(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []
    
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            
            if line:  # If the line is not empty
                parts = line.split()  # Split the line into parts
                
                if len(parts) >= 2:  # Make sure we have at least a word and a tag
                    word = parts[0]  # Word
                    tag = parts[1]   # NER tag
                    
                    # Optionally handle more columns (e.g., chunk, NER type) if they exist
                    sentence.append(word)
                    label.append(tag)
                
            else:
                # When an empty line is encountered, it means the sentence ends
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []  # Reset for the next sentence
                    label = []
        
        # Append the last sentence if the file doesn't end with a blank line
        if sentence:
            sentences.append(sentence)
            labels.append(label)
    
    return sentences, labels

def sent2feats(sentence):
    from nltk import pos_tag

    feats = []
    # Ensure that sentence is a list of words and not a list of sentences
    sen_tags = pos_tag(sentence)  # Perform POS tagging on the sentence
    
    for i in range(0, len(sentence)):
        word = sentence[i]
        wordfeats = {}
        
        # POS tag features: current tag, previous and next 2 tags.
        wordfeats['tag'] = sen_tags[i][1]
        
        # Previous tag feature
        if i == 0:
            wordfeats["prevTag"] = "<S>"
        elif i == 1:
            wordfeats["prevTag"] = sen_tags[0][1]
        else:
            wordfeats["prevTag"] = sen_tags[i - 1][1]
        
        # Next tag feature
        if i == len(sentence) - 2:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
        elif i == len(sentence) - 1:
            wordfeats["nextTag"] = "</S>"
        else:
            wordfeats["nextTag"] = sen_tags[i + 1][1]
        
        feats.append(wordfeats)
    
    return feats

def train_seq(X_train, Y_train, X_dev):
    # Initialize CRF with LBFGS optimization, regularization parameters, and max iterations
    crf = CRF(algorithm='lbfgs', c1=0.1, c2=10, max_iterations=50)
    
    # Train the CRF model
    crf.fit(X_train, Y_train)
    
    # Get the list of labels the model is predicting
    labels = list(crf.classes_)
    
    # Predict the labels for the dev (validation) set
    y_pred = crf.predict(X_dev)
    return y_pred