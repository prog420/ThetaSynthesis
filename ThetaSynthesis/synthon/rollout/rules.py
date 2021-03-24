# -*- coding: utf-8 -*-
#
#  Copyright 2021 Alexander Sizov <murkyrussian@gmail.com>
#  Copyright 2021 Ramil Nugmanov <nougmanoff@protonmail.com>
#  This file is part of ThetaSynthesis.
#
#  ThetaSynthesis is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.
#
from CGRtools import QueryContainer, ReactionContainer
from CGRtools.periodictable import ListElement
from CGRtools.containers.bonds import QueryBond


rules = []

# acyl group addition
q = QueryContainer()
q.add_atom('C', neighbors=1)
q.add_atom('C')
q.add_atom('O')
q.add_atom(ListElement(['N', 'O']))
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 1)

p = QueryContainer()
p.add_atom('A', _map=4)

r = ReactionContainer([q], [p])
rules.append(r)

# acyl group addition with aromatic carbon's case
q = QueryContainer()
q.add_atom('C')
q.add_atom('C')
q.add_atom('O')
q.add_atom('C', hybridization=4)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 1)

p = QueryContainer()
p.add_atom('C', _map=4)

r = ReactionContainer([q], [p])
rules.append(r)

# aryl nitro reduction
q = QueryContainer()
q.add_atom('N', neighbors=1)
q.add_atom('C', hybridization=4, heteroatoms=1)
q.add_bond(1, 2, 1)

p = QueryContainer()
p.add_atom('N', charge=1)
p.add_atom('C')
p.add_atom('O', charge=-1)
p.add_atom('O')
p.add_bond(1, 2, 1)
p.add_bond(1, 3, 1)
p.add_bond(1, 4, 2)

r = ReactionContainer([q], [p])
rules.append(r)

# aryl nitration
q = QueryContainer()
q.add_atom('N', charge=1)
q.add_atom('C', hybridization=4, heteroatoms=(1, 2))
q.add_atom('O', charge=-1)
q.add_atom('O')
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 1)
q.add_bond(1, 4, 2)

p = QueryContainer()
p.add_atom('C', _map=2)

r = ReactionContainer([q], [p])
rules.append(r)

# Beckmann rearrangement (oxime -> amide)
q = QueryContainer()
q.add_atom('C')
q.add_atom('N')
q.add_atom('O')
q.add_atom('C')
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 2)
q.add_bond(2, 4, 1)

p = QueryContainer()
p.add_atom('C')
p.add_atom('N')
p.add_atom('O')
p.add_atom('C')
p.add_bond(1, 2, 2)
p.add_bond(2, 3, 1)
p.add_bond(1, 4, 1)

r = ReactionContainer([q], [p])
rules.append(r)

# aldehydes or ketones into oxime reaction
q = QueryContainer()
q.add_atom('C', hybridization=2)
q.add_atom('N')
q.add_atom('O', hydrogens=1)
q.add_bond(1, 2, 2)
q.add_bond(2, 3, 1)

p = QueryContainer()
p.add_atom('C')
p.add_atom('O')
p.add_bond(1, 2, 2)

r = ReactionContainer([q], [p])
rules.append(r)

# addition of halogen atom into phenol ring (orto)
q = QueryContainer()
q.add_atom(ListElement(['O', 'N']), neighbors=1)
q.add_atom('C')
q.add_atom('C')
q.add_atom(ListElement(['Cl', 'F', 'Br', 'I']))
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 1)

p = QueryContainer()
p.add_atom('A')
p.add_atom('C')
p.add_atom('C')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)

r = ReactionContainer([q], [p])
rules.append(r)

# addition of halogen atom into phenol ring (para)
q = QueryContainer()
q.add_atom(ListElement(['O', 'N']), neighbors=1)
q.add_atom('C')
q.add_atom('C')
q.add_atom('C')
q.add_atom('C')
q.add_atom(ListElement(['Cl', 'F', 'Br', 'I']))
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 4)
q.add_bond(4, 5, 4)
q.add_bond(5, 6, 1)

p = QueryContainer()
p.add_atom('A')
p.add_atom('C')
p.add_atom('C')
p.add_atom('C')
p.add_atom('C')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)
p.add_bond(3, 4, 4)
p.add_bond(4, 5, 4)

r = ReactionContainer([q], [p])
rules.append(r)

# Williamson reaction on phenol
q = QueryContainer()
q.add_atom('C', hybridization=4)
q.add_atom('O')
q.add_atom('C', hybridization=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p1 = QueryContainer()
p1.add_atom('C')
p1.add_atom('O')
p1.add_bond(1, 2, 1)

p2 = QueryContainer()
p2.add_atom('C', _map=3)

r = ReactionContainer([q], [p1, p2])
rules.append(r)

# hard reduction of ketones
q = QueryContainer()
q.add_atom('C', hybridization=4)
q.add_atom('C', hybridization=1, neighbors=2, heteroatoms=0)
q.add_bond(1, 2, 1)

p = QueryContainer()
p.add_atom('C')
p.add_atom('C')
p.add_atom('O')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 2)

r = ReactionContainer([q], [p])
rules.append(r)

# Buchwald-Hartwig amination for arylamine case
q = QueryContainer()
q.add_atom('N', heteroatoms=0, hybridization=1, neighbors=(2, 3))
q.add_atom('C', hybridization=4)
q.add_bond(1, 2, 1)

p1 = QueryContainer()
p1.add_atom('C', _map=2)
p1.add_atom('Cl')
p1.add_bond(2, 3, 1)

p2 = QueryContainer()
p2.add_atom('N')

r = ReactionContainer([q], [p1, p2])
rules.append(r)

# imidazole imine atom's alkylation
q = QueryContainer()
q.add_atom('N')
q.add_atom('C')
q.add_atom('N')
q.add_atom('C', hybridization=1)
q.add_bond(1, 2, 4)
q.add_bond(2, 3, 4)
q.add_bond(1, 4, 1)

p1 = QueryContainer()

p1.add_atom('N')
p1.add_atom('C')
p1.add_atom('N')
p1.add_bond(1, 2, 4)
p1.add_bond(2, 3, 4)

p2 = QueryContainer()
p2.add_atom('C', _map=4)
p2.add_atom('Cl')
p2.add_bond(4, 5, 1)

r = ReactionContainer([q], [p1, p2])
rules.append(r)


__all__ = ['rules']
