#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file      movie_recommendations.py
author    Ryan Aubrey <rma7qb@virginia.edu>
version   1.0
date      May 30, 2017


brief     Use four different loss functions in four models to recommend movies given movies a user already likes
usage     python movie_recommendations.py
"""

import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM

#Fetch the movie data, 100000 ratings
data = fetch_movielens(min_rating=4.0)


print(repr(data['train']))
print(repr(data['test']))

#warp = weighted approximate rank pairwise
model_warp = LightFM(loss='warp')

#logistic loss function
model_logistic = LightFM(loss='logistic')

#Bayesian Personalised Ranking
model_bpr = LightFM(loss='bpr')

#kth-order statistical loss
model_warp_kos = LightFM(loss='warp-kos')



model_warp.fit(data['train'], epochs=30, num_threads=2)
model_logistic.fit(data['train'], epochs=30, num_threads=2)
model_bpr.fit(data['train'], epochs=30, num_threads=2)
model_warp_kos.fit(data['train'], epochs=30, num_threads=2)


def sample_recommendation(model, data, user_ids):

	#Number of users and movies in training set
	n_users, n_items = data['train'].shape

	#Generate recommendations
	for user_id in user_ids:

		#Movies they already liked
		known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]

		#movies our prediction says they will like
		scores = model.predict(user_id, np.arange(n_items))

		#rank them
		top_items = data['item_labels'][np.argsort(-scores)]

		#Return Results
		print("User %s" % user_id)
		print("        Known Positives:")

		#Print top three movies they liked
		for positive in known_positives[:3]:
			print("                    %s" % positive)

		print("        Our Recommended:")

		#Print our recommendation of three movies they would like
		top_found = 0
		for item in top_items:

			#Only want to give three top
			if top_found == 3:
				break

			#Do not want to conflict with already known favorites
			if item not in known_positives[:3]:
				print("                    %s" % item)
				top_found += 1

			#It was in the favorites, ignore and move on
			else:
				pass

print("***")
sample_recommendation(model_warp, data, [3, 25, 44])
print("***")
sample_recommendation(model_bpr, data, [3, 25, 44])
print("***")
sample_recommendation(model_logistic, data, [3, 25, 44])
print("***")
sample_recommendation(model_warp_kos, data, [3, 25, 44])
print("***")

