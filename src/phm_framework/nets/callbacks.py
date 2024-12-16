from phm_framework.nets.metrics import FewShotMetric

def fewshot_extra_callbacks(context):
    model = context['model']
    val_few_shot_gen = context['data_generators']['few_shot_val']

    return [FewShotMetric((val_few_shot_gen,), model)]