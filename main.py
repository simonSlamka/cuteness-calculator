import argparse
from cuteness_calculator import CutenessCalculator

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Calculate the cuteness of a face in an image.")
    parser.add_argument("imgPath", help="The path to the image file.")
    parser.add_argument("predictorPath", help="The path to the dlib shape predictor file.")
    parser.add_argument("directoryPath", help="The path to the directory of images.")
    args = parser.parse_args()

    cutenessCalculator = CutenessCalculator(args.predictorPath)

    minValues, maxValues = cutenessCalculator.calculate_feature_ranges(args.directoryPath, args.imgPath)

    cutenessScore = cutenessCalculator.calculate_cuteness(args.imgPath, minValues, maxValues)

    if cutenessScore is not None:
        print("Cuteness score: {:.2f}".format(cutenessScore))

if __name__ == "__main__":
    main()