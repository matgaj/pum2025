import numpy as np
import random
import music21 as m21
from datetime import datetime
import os
from time import sleep

random.seed(69) # ustawiamy, jeśli chcemy mieć powtarzalne wyniki
BITS_PER_NOTE = 4 #liczba bitów, na których zapisana będzie wysokość pojedynczej nuty


def generate_genome(length):
    """
    Generowanie chromosomu osobnika.
    Funkcja tworzy wektor o długości length, w którym będą znajdowały się wartości 0 i 1 w losowej kolejności.
    Możesz użyć funkcji random.choices lub dowolnej innej.
    """
    genome = random.choices([0,1], k=length)

    return genome


def generate_population(size, genome_length):
    """
    Tworzenie populacji.
    size - liczba osobników w populacji
    genome_length - długość chromosomu każdego osobnika
    Wywołaj funkcję generate_genome tak, by uzyskać osobniki.
    """
    population = [generate_genome(genome_length) for _ in range(size)]

    return population


def single_point_crossover(parent1, parent2):
    """
    Krzyżowanie osobników.
    Funkcja przyjmuje jako argumenty dwa chromosomy, które mają być skrzyżowane.
    1. jeżeli długości chromosomów są różne, funkcja zwraca błąd - to już jest zaimplementowane
    2. jeżeli w chromosomie jest tylko jeden gen, krzyżowanie nie jest możliwe 
    i funkcja powinna zwrócić chromosomy rodziców bez zmian
    3. jeżeli chromosomy zawierają co najmniej 2 geny, funkcja przeprowadza krzyżowanie - 
    chromosomy mają być "przecięte" w losowym miejscu i sklejone tak, by jeden potomek 
    miał pierwszą część od rodzica 1, drugą od rodzica 2, a drugi na odwrót.
    """

    if len(parent1) != len(parent2):
        raise ValueError("Chromosomy rodziców muszą mieć taką samą długość")
    
    if len(parent1) <= 1:
        return parent1, parent2
    
    crossover_point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]

    return offspring1, offspring2

def multi_point_crossover(parent1, parent2, num_points=2):

    if len(parent1) != len(parent2):
        raise ValueError("Chromosomy rodziców muszą mieć taką samą długość")
    
    if len(parent1) < num_points:
        return parent1, parent2
    
    '''
    Losowanie liczby punktów krzyżowania.
    Zapewnienie, że punkty są unikalne i posortowane.
    Tworzenie potomków poprzez naprzemienne kopiowanie segmentów między punktami krzyżowania.
    '''
    
    crossover_points = sorted(random.sample(range(1, len(parent1)), num_points))
    offspring1, offspring2 = [], []
    last_point = 0
    for i, point in enumerate(crossover_points + [len(parent1)]):
        if i % 2 == 0:
            offspring1.extend(parent1[last_point:point])
            offspring2.extend(parent2[last_point:point])
        else:
            offspring1.extend(parent2[last_point:point])
            offspring2.extend(parent1[last_point:point])
        last_point = point

    return offspring1, offspring2


def mutation(genome, num=1, probability=0.5):
    """
    genome - chromosom
    num - liczba potencjalnych mutacji (domyślnie: 1)
    probability - prawdopodobieństwo wystąpienia mutacji (domyślnie: 0.5)
    Funkcja zwraca chromosom po mutacjach.

    W wyniku mutacji 0 zamienia się na 1, a 1 zamienia się na 0.
    Mutacji ulegają losowe geny (elementy wektora genome).
    """
    mutated_genome = genome.copy()
    for _ in range(num):
        if random.random() < probability:
            mutation_index = random.randint(0, len(mutated_genome) - 1)
            mutated_genome[mutation_index] = 1 - mutated_genome[mutation_index]

    return mutated_genome

def reorder_mutation(genome, num=1, probability=0.5):
    # zamienia miejscami dwa losowe geny
    mutated_genome = genome.copy()
    for _ in range(num):
        if random.random() < probability:
            index1 = random.randint(0, len(mutated_genome) - 1)
            index2 = random.randint(0, len(mutated_genome) - 1)
            mutated_genome[index1], mutated_genome[index2] = mutated_genome[index2], mutated_genome[index1]

    return mutated_genome


def get_scale(key, scale, octave):
    # zwraca listę dźwięków należących do wybranej tonacji
    if scale == "major":
        return m21.scale.MajorScale(key).getPitches(key+str(octave))
    elif scale == "minor":
        return m21.scale.MinorScale(key).getPitches(key+str(octave))
    # więcej dostępnych skal w dokumentacji:
    # https://www.music21.org/music21docs/moduleReference/moduleScale.html


def int_from_bits(bits):
    # Konwersja ciągu bitów na wartość całkowitą
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))


def genome_to_melody(genome, num_bars, num_notes, key, scale, octave):
    # Konwersja ciągów bitowych na nuty lub pauzy  

    notes = [genome[i * BITS_PER_NOTE : i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]

    scl = get_scale(key, scale, octave)
    melody = m21.stream.Stream()

    for note in notes:
        integer = int_from_bits(note)
        # wartości 0-7 to indeksy dźwięków w gamie, pozostałe - pauzy
        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody.append(m21.note.Rest(type='quarter'))
        else:
            if (  len(melody) > 0 
                  and melody[-1].isNote 
                  and melody[-1].nameWithOctave == scl[integer].nameWithOctave
                ):
                # jeśli wysokość się powtarza, przedłużamy poprzednią nutę
                d = m21.duration.Duration()
                d.addDurationTuple(melody[-1].duration.type)
                d.addDurationTuple('quarter')
                d.consolidate()
                melody[-1].duration.type = d.type
            else:
                melody.append(m21.note.Note(scl[integer], type='quarter'))

    return melody


def select_pair(population, fitness_func, population_fitness):
    # wybór pary rodziców na podstawie wartości funkcji oceny - 
    # wyższa ocena zwiększa prawdopodobieństwo wylosowania danego osobnika
    return random.choices(
        population=population,
        weights=[fitness_func(genom, population_fitness) for genom in population],
        k=2
    )


def fitness_lookup(genome, population_fitness):
    for e in population_fitness:
        # if e[0] == genome:  # dla genomów zapiasanych jako lista
        if np.array_equal(e[0], genome): # dla genomów zapisanych jako np.array
            return e[1]
    return 0


def fitness(genome, num_bars, num_notes, key, scale, octave):
    # Funkcja przystosowania - w naszym przypadku pyta użytkownika o ocenę melodii

    melody = genome_to_melody(genome, num_bars, num_notes, key, scale, octave)
    melody.show('midi') # odgrywanie melodii
    # melody.show('text') # wypisanie zawartości strumienia (sekwencji nut i pauz)

    rating = input("Ocena (0-5)")
    sleep(1)

    try:
        rating = int(rating)
    except ValueError:
        print("Nierozpoznana wartość, przypisano 0")
        rating = 0

    return rating


def save_genome_to_midi(filename, genome, num_bars, num_notes, key, scale, octave, bpm):
    # Zapis wygenerowanych przebiegów do plików MIDI
    
    melody = genome_to_melody(genome, num_bars, num_notes, key, scale, octave)
    melody.insert(0, m21.tempo.MetronomeMark(bpm))
    melody.insert(0, m21.meter.base.TimeSignature(f"{num_notes}/{4}"))

    if not filename.endswith(".mid"):
        filename += ".mid"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    melody.write('midi', filename)


def run_evolution(parents, mutation1, mutation2, num_mutations=2, mutation_probability=0.5):
    # Funkcja przeprowadza krzyżowanie i mutacje, zwraca nowe osobniki
    num_crossover_points = random.randint(1, 4)
    offspring1, offspring2 = multi_point_crossover(parents[0], parents[1], num_crossover_points)
    if random.random() < 0.5:
        offspring1 = mutation1(offspring1, num=num_mutations, probability=mutation_probability)
    else:
        offspring1 = mutation2(offspring1, num=num_mutations, probability=mutation_probability)
        
    return offspring1, offspring2


def main(num_bars, num_notes, key, scale, octave, population_size, num_mutations, mutation_probability, bpm):
    """
    Główna pętla - losowa inicjalizacja populacji, następnie przeprowadzana
    jest ocena osobników i ewolucja dopóki użytkownik nie zakończy programu
    """

    folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    population_id = 0
    population = [generate_genome(num_bars * num_notes * BITS_PER_NOTE) for _ in range(population_size)]
    print("Inicjalizacja pierwszej populacji zakończona")

    running = True
    while running:
        random.shuffle(population)

        # każdemu genomowi przyporządkowujemy ocenę
        population_fitness = [(genome, fitness(genome, num_bars, num_notes, key, scale, octave)) for genome in population]

        # sortujemy populację od najlepszych osobników
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)
        population = [e[0] for e in sorted_population_fitness]

        # do kolejnej generacji bierzemy dwa najlepsze osobniki
        next_generation = population[:2]
        # oraz dodajemy potomków:
        for _ in range(int(len(population) / 2) - 1):
            parents = select_pair(population, fitness_lookup, population_fitness)
            offspring1, offspring2 = run_evolution(parents=parents,
                                                   mutation1=mutation,
                                                   mutation2=reorder_mutation,
                                                   num_mutations=num_mutations,
                                                   mutation_probability=mutation_probability)
            next_generation += [offspring1, offspring2]

        print("Zapisywanie osobników do plików MIDI")
        for i, genome in enumerate(population):
            filename = f"{folder}/{population_id}/{key}{scale}-pop{i}.mid"
            save_genome_to_midi(filename, genome, num_bars, num_notes, key, scale, octave, bpm)

        print(f"Populacja nr {population_id} zakończona")

        running = input("Czy kontynuować i przejść do kolejnej generacji? [Y/n]") != "n"
        population = next_generation
        population_id += 1
