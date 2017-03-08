# flaredownDiagnosisApi

Flask/Swagger API for diagnosis of chronic illness. Based off of the data from Flaredown.com, and intended for their use.  I won't be adding the data to this repo, you'll need to look elsewhere if you want to run train locally.

### Why a diagnosis engine?

This certainly isn't the world's first attempt at diagnosis through machine learning.  The novelty comes from the fact that we aren't just taking a list of symptoms that a user has deemed relevant, and then guessing a single condition.  Instead this will take the users symptoms and guess a list of concurrent conditions.  This is inspired by the Flaredown data, which showed us that Flaredown users log an average of 8 conditions at once.

Of course, trying to predict a set of conditions is harded.  But my hope is, it's also a more realistic way of looking at illness, especially chronic illness.  I also have hopes that this engine will be able to detect conditions that have gone undiagnosed due to having 'fallen between' other conditions, and hopefully spur a useful conversation with the user's doctor.

### Algorithm

It uses PCA and Ridge Regression.
Algorithm is based on the exploratory work down in the flaredown_data repo.  Which in turn borrows a lot of ideas from David Thaler in his Greek Media project (also available on Github)
