from visualdl.server import app


list1 = [
         './EndEffectorPositioningURSim-v0_100_0', './EndEffectorPositioningURSim-v0_101_0',
         './EndEffectorPositioningURSim-v0_102_0', './EndEffectorPositioningURSim-v0_103_0',
         './EndEffectorPositioningURSim-v0_104_0'
         ]




if __name__ == '__main__':
    app.run(logdir=list1)
