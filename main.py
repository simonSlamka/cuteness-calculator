import argparse
from cuteness_calculator import CutenessCalculator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate the cuteness of a face in an image.")
    parser.add_argument('imgPath', help="The path to the image file.")
    parser.add_argument('predictorPath', help="The path to the dlib shape predictor file.")
    args = parser.parse_args()

    cutenessCalculator = CutenessCalculator(args.predictorPath)

    cutenessScore = cutenessCalculator.calculate_cuteness(args.imgPath)

    print("Cuteness score: {:.2f}".format(cutenessScore))

if __name__ == "__main__":
    main()