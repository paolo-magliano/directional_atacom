from air_hockey_challenge.constraints import ConstraintList

class ConstraintCollection(ConstraintList):
    def output_dim(self):
        return sum(constr.output_dim for constr in self.constraints.values())
