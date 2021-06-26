from src import get_config, train, evaluate, test

# Get config from conf.yaml
conf = get_config('./conf/testing.yaml')

task = conf['task']
if task == 'training':
    train(conf)
elif task == 'evaluation':
    evaluate(conf)
elif task == 'testing':
    test(conf)
else:
    print('Task not supported.')
    exit()
