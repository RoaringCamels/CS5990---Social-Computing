package com.example.agents;

import java.util.ArrayList;
import java.util.List;

import repast.simphony.context.Context;
import repast.simphony.engine.schedule.ScheduledMethod;
import repast.simphony.random.RandomHelper;
import repast.simphony.util.ContextUtils;
import repast.simphony.space.continuous.ContinuousSpace;

public class Person {
    public enum State {
        SUSCEPTIBLE, INFECTED, RECOVERED
    }

    private State state;
    private double beta; // Infection rate
    private double gamma; // Recovery rate
    private double infectionTime;
    private static final double INFECTION_DURATION = 10.0;
    private double x,y;

    public Person(double beta, double gamma) {
        this.state = State.SUSCEPTIBLE;
        this.beta = beta;
        this.gamma = gamma;
        this.infectionTime = -1;
        this.x = RandomHelper.nextDouble()*100;
        this.y = RandomHelper.nextDouble()*100;
    }

    public State getState() {
        return state;
    }

    public void infect() {
        if (state == State.SUSCEPTIBLE) {
            state = State.INFECTED;
            infectionTime = 0;
        }
    }
    
    public boolean tryInfect() {
        return state == State.INFECTED && Math.random() < beta;
    }

    @ScheduledMethod(start = 1, interval = 1)
    public void step() {
        if (state == State.INFECTED) {
            infectionTime++;
            if (infectionTime >= INFECTION_DURATION) {
                if (RandomHelper.nextDouble() < gamma) {
                    state = State.RECOVERED;
                    infectionTime = -1;
                }
            }
        }

        // INFECTION SPREAD MECHANICS
        if (state == State.INFECTED) {
            // Get current context and continuous space
            Context<Object> context = ContextUtils.getContext(this);
            ContinuousSpace<Object> space =
                    (ContinuousSpace<Object>) context.getProjection("space");

            // Find nearby agents (distance-based interaction)
            List<Person> neighbors = getNeighbors(space, this, 2);

            for (Person neighbor : neighbors) {
                // Attempt to infect susceptible neighbors
                if (neighbor.state == State.SUSCEPTIBLE) {
                    if (RandomHelper.nextDouble() < beta) {
                        neighbor.infect();
                    }
                }
            }
        }
    }

 // Get neighbors within a certain radius in continuous space
    public static List<Person> getNeighbors(ContinuousSpace<Object> space, Person agent, double radius) {
        List<Person> neighbors = new ArrayList<>();
        // Get all agents in the context
        Context<Object> context = ContextUtils.getContext(agent);
        for (Object obj : context.getObjects(Person.class)) {
            if (obj != agent) { // Exclude the current agent
                Person neighbor = (Person) obj;
                // Calculate distance between the agents
                double distance = Math.sqrt(Math.pow(agent.x - neighbor.x, 2) + Math.pow(agent.y - neighbor.y, 2));
                if (distance <= radius) {
                    neighbors.add(neighbor);
                }
            }
        }
        return neighbors;
    }
}
