


import org.tensorflow.*;
import java.util.List;

public static void main(String[] args) {
    SavedModelBundle b = SavedModelBundle.load("./model", "mytag");
    Session tfSession = b.session();
    Operation operationAdd = b.graph().operation("predict");
    Output output = new Output(operationAdd, 0);
    float[][] a = new float[1][784];
    a[0] = new float[]{0f,0f,0f,0.592157f,0.592157f,...};
    Tensor input_x = Tensor.create(a);
    List<Tensor> out = tfSession.runner().feed("input_x", input_x).fetch(output).run();
    for (Tensor s : out) {
        float[][] t = new float[1][10];
        s.copyTo(t);
        for (float i : t[0])
            System.out.println(i);
    }
}