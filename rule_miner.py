import itertools
import pandas as pd


class RuleMiner(object):

    def __init__(self, support_t, confidence_t):
        """Class constructor for RuleMiner
        Arguments:
            support_t {int} -- support threshold for the dataset
            confidence_t {int} -- confidence threshold for the dataset
        """
        self.support_t = support_t
        self.confidence_t = confidence_t

    def get_support(self, data, itemset):
        """Returns the support for an itemset."""
        if isinstance(itemset, str):
            itemset = [itemset]
        return data[itemset].astype(bool).all(axis=1).sum()

    def merge_itemsets(self, itemsets):
        """Merge itemsets to create candidate itemsets of higher length."""
        new_itemsets = set()
        if not itemsets:
            return []
            
        itemsets = [frozenset(i) for i in itemsets]
        k = len(next(iter(itemsets)))
        
        for i in range(len(itemsets)):
            for j in range(i + 1, len(itemsets)):
                if sorted(itemsets[i])[:k-1] == sorted(itemsets[j])[:k-1]:
                    merged = itemsets[i].union(itemsets[j])
                    if len(merged) == k + 1:
                        new_itemsets.add(merged)
        return [list(i) for i in new_itemsets]

    def get_rules(self, itemset):
        """Generate all possible association rules from an itemset."""
        if len(itemset) <= 1:
            return []
    
        rules = []
        for i in range(1, len(itemset)):
            for lhs in itertools.combinations(itemset, i):
                lhs = list(lhs)
                rhs = list(set(itemset) - set(lhs))
                rules.append([lhs, rhs])
        return rules

    def get_frequent_itemsets(self, data):
        """Returns frequent itemsets meeting the support threshold."""
        itemsets = [[col] for col in data.columns]
        all_frequent_itemsets = []
        
        while itemsets:
            new_itemsets = []
            for itemset in itemsets:
                support = self.get_support(data, itemset)
                if support >= self.support_t:
                    new_itemsets.append(itemset)
                    all_frequent_itemsets.append(itemset)
            
            itemsets = self.merge_itemsets(new_itemsets)
        
        return all_frequent_itemsets

    def get_confidence(self, data, rule):
        """Calculate confidence for a rule."""
        X = rule[0]
        Y = rule[1]
        if not isinstance(X, list):
            X = [X]
        if not isinstance(Y, list):
            Y = [Y]     
        
        support_union = self.get_support(data, X + Y)
        support_X = self.get_support(data, X)
        
        return support_union / support_X if support_X != 0 else 0.0

    def get_association_rules(self, data):
        """Returns association rules as a DataFrame with columns:
        Antecedent, Consequent, and Confidence.
        """
        itemsets = self.get_frequent_itemsets(data)
        rules_data = []
        
        for itemset in itemsets:
            if len(itemset) >= 2:
                rules = self.get_rules(itemset)
                for rule in rules:
                    if self.get_support(data, rule[0] + rule[1]) >= self.support_t:
                        confidence = self.get_confidence(data, rule)
                        if confidence >= self.confidence_t:
                            # Convert lists to strings for better readability
                            antecedent = ', '.join(rule[0])
                            consequent = ', '.join(rule[1])
                            rules_data.append({
                                'Antecedent': antecedent,
                                'Consequent': consequent,
                                'Confidence': confidence
                            })
        
        # Create DataFrame and sort by confidence (descending)
        rules_df = pd.DataFrame(rules_data)
        if not rules_df.empty:
            rules_df = rules_df.sort_values('Confidence', ascending=False)
            rules_df['Confidence'] = rules_df['Confidence'].round(3)  # Round to 3 decimals
        
        return rules_df