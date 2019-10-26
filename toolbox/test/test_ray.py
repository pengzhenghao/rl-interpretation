import ray

if __name__ == '__main__':
    def f(mb):
        ray.init(object_store_memory=mb * 1024 * 1024)
        print("MB: {}, Resource: {}, object memory: {}\n".format(
            mb, ray.available_resources(),
            ray.available_resources()['object_store_memory']*50))
        ray.shutdown()


    f(80)
    f(100)
    f(150)
    f(200)
    f(300)
    f(390)
