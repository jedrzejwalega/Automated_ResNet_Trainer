import input_data
import train_test
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    args = input_data.get_input()
    input_data.download_mnist(args.path)
    train_images, train_labels = input_data.load_mnist(args.path)
    test_images, test_labels = input_data.load_mnist(args.path)
    program = train_test.RunManager(learning_rates=args.learning_rates, 
                                    epochs=args.epochs,
                                    shuffle=args.shuffle, 
                                    find_lr=args.find_lr, 
                                    batch_size=args.batch_size,
                                    gamma=args.gamma)
    program.model_params(10)
    program.pass_datasets((train_images, train_labels), (test_images, test_labels))
    program.train()
    if args.test:
        program.test()

main()