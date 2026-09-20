import csv
import itertools
import sys

PROBS = {
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },
    "trait": {
        2: {
            True: 0.65,
            False: 0.35
        },
        1: {
            True: 0.56,
            False: 0.44
        },
        0: {
            True: 0.01,
            False: 0.99
        }
    },
    "mutation": 0.01
}


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    names = set(people)
    for have_trait in powerset(names):
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    normalize(probabilities)

    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a CSV file into a dictionary.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return the joint probability of a given configuration of
    genes and traits for everyone in the family tree.
    """
    joint_prob = 1.0

    for person in people:
        # Determine how many gene copies this person has in this configuration
        gene_count = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )
        has_trait = person in have_trait

        mother = people[person]["mother"]
        father = people[person]["father"]

        # If no parents, use unconditionally known population probabilities
        if mother is None and father is None:
            gene_prob = PROBS["gene"][gene_count]
        else:
            # Helper to get the probability of a parent passing on the gene
            def pass_prob(parent_name):
                parent_genes = (
                    2 if parent_name in two_genes else
                    1 if parent_name in one_gene else
                    0
                )
                if parent_genes == 2:
                    return 1 - PROBS["mutation"]
                elif parent_genes == 1:
                    return 0.5
                else:
                    return PROBS["mutation"]

            p_mother = pass_prob(mother)
            p_father = pass_prob(father)

            # Combine inheritance probabilities based on child's gene count
            if gene_count == 2:
                gene_prob = p_mother * p_father
            elif gene_count == 1:
                gene_prob = (p_mother * (1 - p_father)) + ((1 - p_mother) * p_father)
            else:
                gene_prob = (1 - p_mother) * (1 - p_father)

        # Factor in the probability of displaying the trait
        trait_prob = PROBS["trait"][gene_count][has_trait]

        # Accumulate probabilities across all individuals
        joint_prob *= gene_prob * trait_prob

    return joint_prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add a new joint probability p to the running probability totals.
    """
    for person in probabilities:
        gene_count = (
            2 if person in two_genes else
            1 if person in one_gene else
            0
        )
        has_trait = person in have_trait

        probabilities[person]["gene"][gene_count] += p
        probabilities[person]["trait"][has_trait] += p


def normalize(probabilities):
    """
    Normalize probability distributions so that gene totals sum to 1
    and trait totals sum to 1.
    """
    for person in probabilities:
        # Normalize gene probabilities
        gene_total = sum(probabilities[person]["gene"].values())
        if gene_total > 0:
            for gene in probabilities[person]["gene"]:
                probabilities[person]["gene"][gene] /= gene_total

        # Normalize trait probabilities
        trait_total = sum(probabilities[person]["trait"].values())
        if trait_total > 0:
            for trait in probabilities[person]["trait"]:
                probabilities[person]["trait"][trait] /= trait_total


if __name__ == "_main_":
    main()
