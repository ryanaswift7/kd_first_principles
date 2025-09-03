"""
TODO:
- Teacher class which contains a dict of task names -> ft model locations
    - select task name to load ft model for task
- have an individual training function, and also one for applying indiv to the
    full task list
- should have create(task=task), init() [make and use task list], save, load
- init should make sure every task variant is trained and/or available
- change LM class to have a dict of task names and model locations
- default to HF for everything and use TL when necessary
"""