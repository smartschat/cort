from cort.coreference.approaches import entity


__author__ = 'martscsn'


class EntityLeftToRightPerceptron(entity.EntityPerceptron):
    def _get_next(self,
                  substructure,
                  anaphor_to_scored_list,
                  anaphor_to_scored_list_with_cost,
                  anaphor_to_cons_scored_list_with_cost,
                  cluster_feats,
                  dynamic_feats,
                  mode):

        if mode == "train":
            list_for_continuation = self._choose_list_for_continuation(
                anaphor_to_scored_list, anaphor_to_scored_list_with_cost, anaphor_to_cons_scored_list_with_cost
            )
        else:
            list_for_continuation = anaphor_to_scored_list

        for mention in substructure.mentions:
            if mention.is_dummy():
                continue

            if mention not in substructure.outgoing_links:
                _, ana_pair, _ = self._get_for_anaphor(
                    mention,
                    list_for_continuation,
                    substructure,
                    cluster_feats,
                    dynamic_feats
                )

                return ana_pair

    def _get_for_anaphor(self, ana, list_to_consider, substructure,
                         cluster_feats, dynamic_feats):
        ana_score, ana_pair, feats = \
            self._get_rescored(
                ana,
                list_to_consider[ana],
                substructure,
                cluster_feats,
                dynamic_feats
            )

        return ana_score, ana_pair, feats

    def _choose_list_for_continuation(self, anaphor_to_scored_list, anaphor_to_scored_list_with_cost, anaphor_to_cons_scored_list_with_cost):
        raise NotImplementedError()


class EntityLeftToRightPerceptronWithGoldRollIn(EntityLeftToRightPerceptron):
    def _choose_list_for_continuation(self, anaphor_to_scored_list, anaphor_to_scored_list_with_cost, anaphor_to_cons_scored_list_with_cost):
        return anaphor_to_cons_scored_list_with_cost


class EntityLeftToRightPerceptronWithLearnedRollIn(EntityLeftToRightPerceptron):
    def _choose_list_for_continuation(self, anaphor_to_scored_list, anaphor_to_scored_list_with_cost, anaphor_to_cons_scored_list_with_cost):
        return anaphor_to_scored_list
