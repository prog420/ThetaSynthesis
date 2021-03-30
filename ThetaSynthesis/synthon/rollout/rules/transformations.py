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


rules = []


def prepare():
    q_ = QueryContainer()
    p_ = QueryContainer()
    rules.append(ReactionContainer((q_,), (p_,)))
    return q_, p_


# aryl nitro reduction
# [C;Za;W1]-[N;D1]>>[O-]-[N+](-[C])=[O]
q, p = prepare()
q.add_atom('N', neighbors=1)
q.add_atom('C', hybridization=4, heteroatoms=1)
q.add_bond(1, 2, 1)

p.add_atom('N', charge=1)
p.add_atom('C')
p.add_atom('O', charge=-1)
p.add_atom('O')
p.add_bond(1, 2, 1)
p.add_bond(1, 3, 1)
p.add_bond(1, 4, 2)


# aryl nitration
# [O-]-[N+](=[O])-[C;Za;W12]>>[C]
q, p = prepare()
q.add_atom('N', charge=1)
q.add_atom('C', hybridization=4, heteroatoms=(1, 2))
q.add_atom('O', charge=-1)
q.add_atom('O')
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 1)
q.add_bond(1, 4, 2)

p.add_atom('C', _map=2)


# Beckmann rearrangement (oxime -> amide)
# [C]-[N;D2]-[C]=[O]>>[O]-[N]=[C]-[C]
q, p = prepare()
q.add_atom('C')
q.add_atom('N', neighbors=2)
q.add_atom('O')
q.add_atom('C')
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 2)
q.add_bond(2, 4, 1)

p.add_atom('C')
p.add_atom('N')
p.add_atom('O')
p.add_atom('C')
p.add_bond(1, 2, 2)
p.add_bond(2, 3, 1)
p.add_bond(1, 4, 1)


# aldehydes or ketones into oxime/imine reaction
# [C;Zd;W1]=[N]>>[C]=[O]
q, p = prepare()
q.add_atom('C', hybridization=2, heteroatoms=1)
q.add_atom('N')
q.add_bond(1, 2, 2)

p.add_atom('C')
p.add_atom('O', _map=3)
p.add_bond(1, 3, 2)


# addition of halogen atom into phenol ring (orto)
# [C](-[Cl,F,Br,I;D1]):[C]-[O,N;Zs]>>[C](-[A]):[C]
q, p = prepare()
q.add_atom(ListElement(['O', 'N']), hybridization=1)
q.add_atom('C')
q.add_atom('C')
q.add_atom(ListElement(['Cl', 'F', 'Br', 'I']), neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 1)

p.add_atom('A')
p.add_atom('C')
p.add_atom('C')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)


# addition of halogen atom into phenol ring (para)
# [C](:[C]:[C]:[C]-[O,N;Zs])-[Cl,F,Br,I;D1]>>[A]-[C]:[C]:[C]:[C]
q, p = prepare()
q.add_atom(ListElement(['O', 'N']), hybridization=1)
q.add_atom('C')
q.add_atom('C')
q.add_atom('C')
q.add_atom('C')
q.add_atom(ListElement(['Cl', 'F', 'Br', 'I']), neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 4)
q.add_bond(4, 5, 4)
q.add_bond(5, 6, 1)

p.add_atom('A')
p.add_atom('C')
p.add_atom('C')
p.add_atom('C')
p.add_atom('C')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)
p.add_bond(3, 4, 4)
p.add_bond(4, 5, 4)


# hard reduction of Ar-ketones
# [C;Za]-[C;D2;Zs;W0]>>[C]-[C]=[O]
q, p = prepare()
q.add_atom('C', hybridization=4)
q.add_atom('C', hybridization=1, neighbors=2, heteroatoms=0)
q.add_bond(1, 2, 1)

p.add_atom('C')
p.add_atom('C')
p.add_atom('O')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 2)


# reduction of alpha-hydroxy pyridine
# [C;W1]:[N;H0;r6]>>[C](:[N])-[O]
q, p = prepare()
q.add_atom('C', heteroatoms=1)
q.add_atom('N', rings_sizes=6, hydrogens=0)
q.add_bond(1, 2, 4)

p.add_atom('C')
p.add_atom('N')
p.add_atom('O')
p.add_bond(1, 2, 4)
p.add_bond(1, 3, 1)


# Reduction of alkene
# [C]-[C;D23;Zs;W0]-[C;D123;Zs;W0]>>[C](-[C])=[C]
q, p = prepare()
q.add_atom('C')
q.add_atom('C', heteroatoms=0, neighbors=(2, 3), hybridization=1)
q.add_atom('C', heteroatoms=0, neighbors=(1, 2, 3), hybridization=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p.add_atom('C')
p.add_atom('C')
p.add_atom('C')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 2)


# Kolbe-Schmitt reaction
# [C](:[C]-[O;D1])-[C](=[O])-[O;D1]>>[C](-[O]):[C]
q, p = prepare()
q.add_atom('O', neighbors=1)
q.add_atom('C')
q.add_atom('C')
q.add_atom('C')
q.add_atom('O', neighbors=1)
q.add_atom('O')
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 4)
q.add_bond(3, 4, 1)
q.add_bond(4, 5, 1)
q.add_bond(4, 6, 2)

p.add_atom('O')
p.add_atom('C')
p.add_atom('C')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 4)


# reduction of carboxylic acid
# [O;D1]-[C;D2]-[C]>>[C]-[C](-[O])=[O]
q, p = prepare()
q.add_atom('C')
q.add_atom('C', neighbors=2)
q.add_atom('O', neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 1)

p.add_atom('C')
p.add_atom('C')
p.add_atom('O')
p.add_atom('O')
p.add_bond(1, 2, 1)
p.add_bond(2, 3, 1)
p.add_bond(2, 4, 2)


# halogenation of alcohols
# [C;Zs]-[Cl,Br;D1]>>[C]-[O]
q, p = prepare()
q.add_atom('C', hybridization=1, heteroatoms=1)
q.add_atom(ListElement(['Cl', 'Br']), neighbors=1)
q.add_bond(1, 2, 1)

p.add_atom('C')
p.add_atom('O', _map=3)
p.add_bond(1, 3, 1)


# Kolbe nitrilation
# [N]#[C]-[C;Zs;W0]>>[Br]-[C]
q, p = prepare()
q.add_atom('C', heteroatoms=0, hybridization=1)
q.add_atom('C')
q.add_atom('N')
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 3)

p.add_atom('C')
p.add_atom('Br', _map=4)
p.add_bond(1, 4, 1)


# Nitrile hydrolysis
# [O;D1]-[C]=[O]>>[N]#[C]
q, p = prepare()
q.add_atom('C')
q.add_atom('O', neighbors=1)
q.add_atom('O')
q.add_bond(1, 2, 1)
q.add_bond(1, 3, 2)

p.add_atom('C')
p.add_atom('N', _map=4)
p.add_bond(1, 4, 3)


# sulfamidation
# [c]-[S](=[O])(=[O])-[N]>>[c]
q, p = prepare()
q.add_atom('C', hybridization=4)
q.add_atom('S')
q.add_atom('O')
q.add_atom('O')
q.add_atom('N', neighbors=1)
q.add_bond(1, 2, 1)
q.add_bond(2, 3, 2)
q.add_bond(2, 4, 2)
q.add_bond(2, 5, 1)

p.add_atom('C')


__all__ = ['rules']
