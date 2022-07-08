use bevy::prelude::Component;

#[derive(Clone, Component, Default)]
pub struct Instances<T> {
    pub values: Vec<T>,
    pub extracted: bool,
}
