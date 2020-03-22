#Fillna Function:
def fill_na(df_new):
    """Iterate through columns , if any nulls replace with 0"""
    for col in df_new.columns:
        if (df_new[col].isnull().any()):
            if (col):
                df_new[col].fillna(0, inplace=True)         
    return df_new
# df_new = fill_na(df_new)

#simply making columnns lowercase
new_cols = [col.lower() for col in df0.columns]
df0.columns = new_cols

#Missing Data Function
def missing_data(df):
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


#wordcloud function
stopwords = set(STOPWORDS)
def show_wordcloud(feature,df1,title="",size=2):
    data = df1.loc[~df1[feature].isnull(), feature].values
    count = (~df1[feature].isnull()).sum()
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40, 
        scale=5,
        random_state=1
    ).generate(str(df))

    fig = plt.figure(1, figsize=(size*4, size*4))
    plt.axis('off')
    fig.suptitle("Prevalent words in {} {} ({} rows)".format(title,feature,count), fontsize=np.sqrt(size)*15)
    fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()

    
# Plot Model Visualisation Fnction
def plot_features(model , figsize):
    """plot feature importances of models"""
    feat_importance = pd.Series(model.feature_importances_ , index = features.columns)
    ax.set_ylabel("features" , size = 16);
    feat_importance.nlargest(10).sort_values().plot(kind = "barh" , figsize = (10 , 5))
    plt.xlabel("Relative Feature Importance For Random Forest");
    plt.title("Feature Importance In Order" , size = 16);
    
    

#Base Fucntion
def base_func(element):
    """train and fit the model"""
    model = element()
    model.fit(X_train , y_train)
    
    """predict"""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    """evaluation"""
    train_accuracy = roc_auc_score(y_train , train_preds)
    test_accuracy = roc_auc_score(y_test , test_preds)
    
    print(str(element))
    print("--------------------------------------------")
    print(f"Training Accuracy: {(train_accuracy * 100) :.4}%")
    print(f"Test Accuracy : {(test_accuracy * 100) :.4}%")
    
    """Store accuracy in a new DataFrame"""
    score_logreg = [element , train_accuracy , test_accuracy]
    print("------------------------------------------------")
    models = pd.DataFrame([score_logreg])    

    
# RUn Model Is My Main Function where I have dumped everything    
def run_model2(model, X_train, y_train,X_test, y_test ):
    model.fit(X_train, y_train)

    """predict"""
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    """evaluation"""
    train_accuracy = roc_auc_score(y_train, train_preds)
    test_accuracy = roc_auc_score(y_test, test_preds)
    report = classification_report(y_test, test_preds)

    """print confusion matrix"""
    cnf_matrix = confusion_matrix(y_test , test_preds)
    print("Confusion Matrix:\n" , cnf_matrix)

    """print reports of the model accuracy"""
    print('Model Scores')
    print("------------------------")
    print(f"Training Accuracy: {(train_accuracy * 100):.4}%")
    print(f"Test Accuracy:     {(test_accuracy * 100):.4}%")
    print("------------------------------------------------------")
    print('Classification Report : \n', report)
    print("-----------------------------------------------------")
    print("Confusion Matrix:\n" , cnf_matrix)
    

    
# Confusion Matrix
def Conf_Matrix(CM , labels = ['pay','default']):
    df0 = pd.DataFrame(data = CM , index = labels, columns = labels)
    df0.index.name = 'True'
    df0.columns.name = "Precision"
    df0.loc['Total'] = df0.sum()
    df0['Total'] = df0.sum(axis = 1)
    return df0

from sklearn.metrics import confusion_matrix
predictions = model.predict(X_test3)


# Print confusion matrix
cnf_matrix = confusion_matrix(y_test3, predictions)
print('Confusion Matrix:\n', cnf_matrix)


# diving deeper into confusion matrix
def conf_mat(actual, predicted):
    cm = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    
    for ind, label in enumerate(actual):
        pred = predicted[ind]
        if label == 1:
            # CASE: TP 
            if label == pred:
                cm['TP'] += 1
            # CASE: FN
            else:
                cm['FN'] += 1
        else:
            # CASE: TN
            if label == pred:
                cm['TN'] += 1
            # CASE: FP
            else:
                cm['FP'] += 1
    return cm

conf_mat(actual, predicted)


# Create the basic matrix
def plot_confusion_matrix(cnf_matrix,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    

    plt.imshow(cnf_matrix,  cmap=plt.cm.Blues) 

"""Add title and axis labels"""    
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""Add appropriate axis scales"""
    class_names = set(y) # Get class labels to add to matrix
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

""" Add labels to each cell"""
    thresh = cnf_matrix.max() / 2. # Used for text coloring below
"""Here we iterate through the confusion matrix and append labels to our visualization"""
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
            horizontalalignment='center',
                 color='white' if cnf_matrix[i, j] > thresh else 'black')

"""Add a legend"""
    plt.colorbar()
    plt.show()

    
# Updated Confusion matrix
def plot_confusion_matrix1(cnf_matrix, classes = class_names ,normalize=False,
                          title='Confusion matrix',
                          class_names = ['Non Default' , 'Default'], 
                          cmap=plt.cm.Blues):
    # Pseudocode/Outline:
    # Print the confusion matrix (optional)
    # Create the basic matrix
    # Add title and axis labels
    # Add appropriate axis scales
    # Add labels to each cell
    # Add a legend
    
#   print(cnf_matrix)

    plt.imshow(cnf_matrix, cmap=cmap)
    
    # Add title and axis labels 
    plt.title('Confusion Matrix') 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = cnf_matrix.max() / 2.
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cnf_matrix[i, j] > thresh else 'black')
    
    # Add a legend
    plt.colorbar()
    plt.show() 

    


# Normalized Confusion Matrix
def plot_Norm_Cnf_matrix(cm, classes,
                          normalize=False,
                          title='Normalized Confusion matrix',
                          cmap=plt.cm.Blues):
    
    # Check if normalize is set to True
    # If so, normalize the raw confusion matrix before visualizing
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, cmap=cmap)
    
    # Add title and axis labels 
    plt.title('Confusion Matrix') 
    plt.ylabel('True label') 
    plt.xlabel('Predicted label')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = cm.max() / 2.
    # Here we iterate through the confusion matrix and append labels to our visualization 
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    # Add a legend
    plt.colorbar()
    plt.show() 

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
     

    
    
 # ROC CURVE
def scores(model,X_train,X_val,y_train3,y_val):
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    train = roc_auc_score(y_train,train_prob)
    val = roc_auc_score(y_val,val_prob)
    print('train:',round(train,2),'test:',round(val,2))
    
# annotation 
def annot(fpr,tpr,thr):
    k=0
    for i,j in zip(fpr,tpr):
        if k %50 == 0:
            plt.annotate(round(thr[k],2),xy=(i,j), textcoords='data')
        k+=1
    
# Plot ROC curve
def roc_plot(model,X_train,y_train,X_val,y_val):
    train_prob = model.predict_proba(X_train)[:,1]
    val_prob = model.predict_proba(X_val)[:,1]
    plt.figure(figsize=(7,7))
    for data in [[y_train, train_prob],[y_val, val_prob]]: # ,[y_test, test_prob]
        fpr, tpr, threshold = roc_curve(data[0], data[1])
        plt.plot(fpr, tpr)
    annot(fpr, tpr, threshold)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.ylabel('TPR (power)')
    plt.xlabel('FPR (alpha)')
    plt.legend(['train','val'])
    plt.show()
    
# Optimized Model
def opt_plots(opt_model):
    opt = pd.DataFrame(opt_model.cv_results_)
    cols = [col for col in opt.columns if ('mean' in col or 'std' in col) and 'time' not in col]
    params = pd.DataFrame(list(opt.params))
    opt = pd.concat([params,opt[cols]],axis=1,sort=False)
    
    plt.figure(figsize=[15,4])
    plt.subplot(121)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_train_score')*100)
    plt.title('ROC_AUC - Training')
    plt.subplot(122)
    sns.heatmap(pd.pivot_table(opt,index='max_depth',columns='min_samples_leaf',values='mean_test_score')*100)
    plt.title('ROC_AUC - Validation')
#     return opt
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    