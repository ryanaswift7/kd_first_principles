
# initialize (finetuned) teacher
# - have a folder with the teacher name
# - subdirs for each task
# - each subdir has one model (i.e. one model per task)

"""
initialize offline projector
- probably need to grab caching script and cache some acts
- only need to cache for teacher and narrow student (i.e. 410m)
- make fully linear and nonlinear (i.e. ReLU) variants -> TODO
"""