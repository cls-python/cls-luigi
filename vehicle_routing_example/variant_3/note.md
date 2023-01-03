# CLS Luigi Filter on Repository
- only non-abstract types end up in the repository (RepoMeta.repository).  
- abstract types can appear in subtypes. Here an inheritance hierarchy could be formed.
  - or better, for all non-abstract types (concrete implementations), find all parent classes where this concrete type can be used.
  - Empty Set on Subtypes is end of inheritance hierarchy.
- small problem: one could implement a inheritance chain where there are classes that are markt as abstract (abstract = True) but inherite from a class that is not abstract (abstract = False). Makes it really hard to filter the repository by just providing concret tasks or abstract tasks and autmatically filter all non  fitting combinators.