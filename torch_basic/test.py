class make_empty_object():
    pass


args = make_empty_object()
args.aa = make_empty_object()
args.aa.cc = 100
print(args.aa.cc)
