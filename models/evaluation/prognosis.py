from collections import OrderedDict
import numpy as np


# Creates an grid with region-of-interest given a range and step size
def create_buckets(x_s, y_s):
	x_min, x_max, x_b = x_s
	y_min, y_max, y_b = y_s

	x = np.linspace(x_min, x_max, x_b)
	x_ranges = list()
	for i_x in range(x_b):
		x_ranges.append((x[i_x], x[i_x+1]))
		if i_x+1 == x_b-1:
			break

	y = np.linspace(y_min, y_max, y_b)
	y_ranges = list()
	for i_y in range(y_b):
		y_ranges.append((y[i_y], y[i_y+1]))
		if i_y+1 == y_b-1:
			break

	zones = list()
	for x_range in x_ranges:
		for y_range in y_ranges:
			zones.append((x_range[0], x_range[1], y_range[0], y_range[1]))

	print('X range:', x_s)
	print('Y range:', y_s)
	print('Number of Buckets:', len(rois))
	return zones

# Classifies a tissue patch assigning it to a region-of-interest. 
def classify_patch(patch_emb, rois):
    x_i, y_i = patch_emb
    for roi_i, roi_range in enumerate(rois):
        x_min, x_max, y_min, y_max = roi_range
        if (x_i < x_max and x_min < x_i) and (y_i < y_max and y_min < y_i):
            return roi_i
    return len(rois)


# Dataset labels, creates a dictionary per patient_id, with keys of 
# 'label' and 'patches', list of indeces with patient patches.
def patient_ids_to_data(labels):
    ids_to_ind = dict()
    for index, fields in enumerate(labels):
        patient_id = fields[0]
        if patient_id not in ids_to_ind:
            ids_to_ind[patient_id] = OrderedDict()
            ids_to_ind[patient_id]['patches'] = list()
            if fields[1] > 5:
                ids_to_ind[patient_id]['label'] = 1
            else:
                ids_to_ind[patient_id]['label'] = 0
        ids_to_ind[patient_id]['patches'].append(index)
    return ids_to_ind


# Method that prepares data for training, takes in the embedding and patient_ids to indeces dict.
# Gives an array with 0-ID, 1-Label, 2:-Features per patient.
def classify_dataset(embedding, ids_to_ind, rois=None, min_patches=12, norm=True):
	print('Minimun number of patches per patient:', min_patches)
	dropped = list()
	n_patients = 0
	for patient in ids_to_ind:
		if len(ids_to_ind[patient]['patches']) < min_patches:
			dropped.append(patient)
			continue
		else:
			n_patients+=1

	if rois is not None:
		print('Using ROIs provided, ID+Label+#ROIs', len(rois)+3)
		patient_features = np.zeros((n_patients, len(rois)+3))
	else: 
		print('No ROIs provided, using encoding as features.')
		patient_features = np.zeros((n_patients, embedding.shape[1]+2))

	print('Dropped patients:', len(dropped), ' IDs:', dropped)
	print('Number of Patients:', patient_features.shape[0])
	print('Number of Features:', patient_features.shape[1])
	ind_dic = 0
	for patient in ids_to_ind:
		patches_patient = ids_to_ind[patient]['patches']
		if len(patches_patient) < min_patches:
			continue
		patient_features[ind_dic, 0] = patient
		patient_features[ind_dic, 1] = ids_to_ind[patient]['label']
        
		if rois is not None:
			for patch in patches_patient:
				roi = classify_patch(embedding[patch], rois)
				patient_features[ind_dic, roi+2] += 1
			if norm:
				total_patient = np.sum(patient_features[ind_dic, 2:])
				patient_features[ind_dic, 2:] = patient_features[ind_dic, 2:]/total_patient
		else:
			patient_features[ind_dic, 2:] = np.mean(embedding[patches_patient, :])
		ind_dic += 1
	return patient_features
    

# Prepares patient features array for training and test, balances both of them so 
# there's no more > 5 years. 
def prepare_data(patient_roi, ratio_training=0.8, display=True):
    l5_patients_ind = np.argwhere(patient_roi[:, 1]==1)[:, 0]
    s5_patients_ind = np.argwhere(patient_roi[:, 1]==0)[:, 0]
    num_train = int(ratio_training*len(s5_patients_ind))
    num_test = len(s5_patients_ind) - num_train

    np.random.shuffle(l5_patients_ind)

    train_l5 = l5_patients_ind[:num_train]
    test_l5 = l5_patients_ind[num_train:num_train+num_test]
    train_s5 = s5_patients_ind[:num_train]
    test_s5 = s5_patients_ind[num_train:]

    train = np.concatenate([train_s5, train_l5])
    test = np.concatenate([test_s5, test_l5])

    np.random.shuffle(train)
    np.random.shuffle(test)

    all_ = np.concatenate([l5_patients_ind, s5_patients_ind])
    all_features = patient_roi[all_, 2:]
    all_labels = patient_roi[all_, 1].astype(np.int)

    train_features = patient_roi[train, 2:]
    train_labels = patient_roi[train, 1].astype(np.int)

    test_features = patient_roi[test, 2:]
    test_labels = patient_roi[test, 1].astype(np.int)
    
    if display:
        print('Larger 5:')
        print('\tTrain:', len(train_l5))
        print('\tTest:', len(test_l5))
        
        print('Smaller 5:')
        print('\tTrain:', len(train_s5))
        print('\tTest:', len(test_s5))
    
    return [train_features, train_labels], [test_features, test_labels], [all_features, all_labels]


# Runs Logistic Regression over train and test set.
def logistic_regression(train, test, all_):
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics

    logisticRegr = LogisticRegression(solver='liblinear')
    logisticRegr.fit(train[0], train[1])

    predictions = logisticRegr.predict(test[0])

    train_score = logisticRegr.score(train[0], train[1])
    test_score = logisticRegr.score(test[0], test[1])
    all_score = logisticRegr.score(all_[0], all_[1])
    cm = metrics.confusion_matrix(test[1], predictions)

    print('Train Accuracy', np.round(train_score,2))
    print('Test Accuracy', np.round(test_score,2))
    print('All Accuracy', np.round(all_score,2))
    print('Confusion matrix:')
    print(cm)


# Pull image from a range of x,y values given the encodings,
# Bucket verifies that the index before to certain list of interest. 
def pull_image_index(x_min, x_max, y_min, y_max, encodings, bucket_ind):
    image_ind = list()
    for index in range(encodings.shape[0]):
        x_i, y_i = encodings[index, :]
        if (x_i < x_max and x_min < x_i) and (y_i < y_max and y_min < y_i) and index in bucket_ind:
            image_ind.append(index)
    return image_ind
