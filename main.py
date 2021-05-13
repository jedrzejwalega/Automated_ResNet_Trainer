
import input_data
import train_test
import cifar10_dataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



def main():
    args = input_data.get_user_input()
    cifar10_dataset.download_cifar10(args.path)
    train_images, train_labels = cifar10_dataset.load_cifar10(args.path)
    test_images, test_labels = cifar10_dataset.load_cifar10(args.path, kind="test")
    program = train_test.RunManager(learning_rates=args.learning_rates, 
                                    epochs=args.epochs,
                                    shuffle=args.shuffle, 
                                    find_lr=args.find_lr, 
                                    batch_size=args.batch_size,
                                    gamma=args.gamma,
                                    gamma_step=args.gamma_step,
                                    momentum=args.momentum,
                                    architectures=args.architecture,
                                    find_gamma_step=args.find_gamma_step,
                                    transform_train=cifar10_dataset.augment_data_train,
                                    transform_valid=cifar10_dataset.augment_data_valid,
                                    comment=args.comment,
                                    initialization=args.initialization)
    program.pass_datasets((train_images, train_labels), (test_images, test_labels))
    program.train()
    if args.test:
        program.test()
    if args.write_model:
        program.write_best_model(args.write_model)

if __name__ == "__main__":
    main()